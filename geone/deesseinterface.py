#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Python module:  'deesseinterface.py'
author:         Julien Straubhaar
date:           jan-2018

Module interfacing deesse for python.
"""

import numpy as np
import sys, os, re, copy
# import multiprocessing

from geone import img, blockdata
from geone.deesse_core import deesse
from geone.img import Img, PointSet
from geone.blockdata import BlockData

# ============================================================================
class SearchNeighborhoodParameters(object):
    """
    Defines search neighborhood parameters:
        radiusMode: (string) radius mode, possible strings:
                        - 'large_default':
                            large radii set according to the size of the SG and
                            the TI(s), and the use of homothethy and/or rotation
                            for the simulation
                            (automatically computed)
                        - 'ti_range_default':
                            search radii set according to the TI(s) variogram
                            ranges, one of the 5 next modes 'ti_range_*' will be
                            used according to the use of homothethy and/or
                            rotation for the simulation
                            (automatically computed)
                        - 'ti_range':
                            search radii set according to the TI(s) variogram
                            ranges, independently in each direction
                            (automatically computed)
                        - 'ti_range_xy':
                            search radii set according to the TI(s) variogram
                            ranges, rx = ry independently from rz
                            (automatically computed)
                        - 'ti_range_xz':
                            search radii set according to the TI(s) variogram
                            ranges, rx = rz independently from ry
                            (automatically computed)
                        - 'ti_range_yz':
                            search radii set according to the TI(s) variogram
                            ranges, ry = rz independently from rx
                            (automatically computed)
                        - 'ti_range_xyz':
                            search radii set according to the TI(s) variogram
                            ranges, rx = ry = rz
                            (automatically computed)
                        - 'manual':
                            search radii rx, ry, rz, are explicitly given

        rx, ry, rz: (floats) radii in each direction
                        (used only if radiusMode is set to 'manual')

        anisotropyRatioMode:
                    (string) anisotropy ratio mode (describing how to set
                        up the anisotropy - i.e. inverse unit distance,
                        ax, ay, az - in each direction), possible strings:
                            - 'one': ax = ay = az = 1
                            - 'radius': ax = rx, ay = ry, az = rz
                            - 'radius_xy': ax = ay = max(rx, ry), az = rz
                            - 'radius_xz': ax = az = max(rx, rz), ay = ry
                            - 'radius_yz': ay = az = max(ry, rz), ax = rx
                            - 'radius_xyz': ax = ay = az = max(rx, ry, rz)
                            - 'manual': ax, ay, az explicitly given
                        if anisotropyRatioMode is set to 'one':
                            isotropic distance - maximal distance for search
                            neighborhood nodes will be equal to the maximum of
                            the search radii
                        if anisotropyRatioMode is set to 'radius':
                            anisotropic distance - nodes at distance one on the
                            border of the search neighborhood, maximal distance
                            for search neighborhood nodes will be 1
                        if anisotropyRatioMode is set to 'radius_*':
                            anisotropic distance - maximal distance for search
                            neighborhood nodes will be 1

        ax, ay, az: (floats) anisotropy (inverse unit distance) in each direction

        angle1, angle2, angle3:
                    (floats) angles (azimuth, dip, plunge in degrees) for rotation
        power:      (float) power for computing weight according to distance
    """

    def __init__(self,
                 radiusMode='large_default',
                 rx=0., ry=0., rz=0.,
                 anisotropyRatioMode='one',
                 ax=0., ay=0., az=0.,
                 angle1=0., angle2=0., angle3=0.,
                 power=0.):
        self.radiusMode = radiusMode
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.anisotropyRatioMode = anisotropyRatioMode
        self.ax = ax
        self.ay = ay
        self.az = az
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
        self.power = power
# ============================================================================

# ============================================================================
class SoftProbability(object):
    """
    Defines probability constraints (for one variable):
        probabilityConstraintUsage:
                    (int) indicates the usage of probability constraints:
                        - 0: no probability constraint
                        - 1: global probability constraints
                        - 2: local probability constraints

        nclass:     (int) number of classes of values
                        (unused if probabilityConstraintUsage == 0)
        classInterval:
                    (list of nclass 2-dimensional array of floats with 2 columns)
                        definition of the classes of values by intervals,
                        classInterval[i] is a (n_i, 2) array a, defining the
                        i-th class as the union of intervals:
                            [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][

        globalPdf:  (1-dimensional array of floats of size nclass)
                        global probability for each class,
                        used when probabilityConstraintUsage == 1

        localPdf:   ((nclass, nz, ny, nx) array of floats) probability for
                        each class, localPdf[i] is the "map defined on the
                        simulation grid (SG)", nx x ny x nz being the dimensions
                        of the SG, used when probabilityConstraintUsage == 2

        localPdfSupportRadius:
                    (1-dimensional array of float size 1) support radius for
                        local pdf, used when probabilityConstraintUsage == 2

        localCurrentPdfComputation:
                    (int) indicates the method used for computing the local
                        current pdf:
                            - 0: "COMPLETE" mode: all the informed nodes in
                                the search neighborhood, and within the support
                                are taken into account
                            - 1: "APPROXIMATE" mode: only the neighboring nodes
                                (used for the search in the TI) within the
                                support are taken into account
                        used when probabilityConstraintUsage == 2

        comparingPdfMethod:
                    (int) indicates  the method used for comparing pdf's:
                            - 0: MAE (Mean Absolute Error)
                            - 1: RMSE (Root Mean Squared Error)
                            - 2: KLD (Kullback Leibler Divergence)
                            - 3: JSD (Jensen-Shannon Divergence)
                            - 4: MLikRsym (Mean Likelihood Ratio (over each
                                    class indicator, symmetric target interval))
                            - 5: MLikRopt (Mean Likelihood Ratio (over each
                                    class indicator, optimal target interval))
                        used when probabilityConstraintUsage > 0

        deactivationDistance:
                    (float) deactivation distance (the probability constraint
                        is deactivated if the distance between the current
                        simulated node and the last node in its neighbors (used
                        for the search in the TI) (distance computed according
                        to the corresponding search neighborhood parameters) is
                        below the given deactivation distance),
                        used when probabilityConstraintUsage > 0

        probabilityConstraintThresholdType:
                    (int) indicates the type of (acceptance) threhsold for
                        pdf's comparison:
                            - 0: constant threshold
                            - 1: dynamic threshold

        constantThreshold:
                    (float) (acceptance) threshold value for pdf's comparison,
                        used when probabilityConstraintUsage > 0 and
                        probabilityConstraintThresholdType == 0

        dynamicThresholdParameters:
                    (1-dimensional array of floats of size 7) parameters for
                        dynamic threshold (used for pdf's comparison),
                        used when probabilityConstraintUsage > 0 and
                        probabilityConstraintThresholdType == 1
    """

    def __init__(self,
                 probabilityConstraintUsage=0,
                 nclass=0,
                 classInterval=None,
                 globalPdf=None,
                 localPdf=None,
                 localPdfSupportRadius=12.,
                 localCurrentPdfComputation=0,
                 comparingPdfMethod=5,
                 deactivationDistance=4.,
                 probabilityConstraintThresholdType=0,
                 constantThreshold=1.e-3,
                 dynamicThresholdParameters=None):
        self.probabilityConstraintUsage = probabilityConstraintUsage
        self.nclass = nclass
        if classInterval is None:
            self.classInterval = classInterval
        else:
            self.classInterval = [np.asarray(ci).reshape(-1,2) for ci in classInterval]

        if globalPdf is None:
            self.globalPdf = None
        else:
            try:
                self.globalPdf = np.asarray(globalPdf, dtype=float).reshape(nclass)
            except:
                print('ERROR: (SoftProbability) field "globalPdf"...')
                return

        if localPdf is None:
            self.localPdf = None
        else:
            self.localPdf = np.asarray(localPdf, dtype=float)

        self.localPdfSupportRadius = localPdfSupportRadius
        self.localCurrentPdfComputation = localCurrentPdfComputation
        self.comparingPdfMethod = comparingPdfMethod
        self.deactivationDistance = deactivationDistance
        self.probabilityConstraintThresholdType = probabilityConstraintThresholdType
        self.constantThreshold = constantThreshold
        self.dynamicThresholdParameters = dynamicThresholdParameters
# ============================================================================

# ============================================================================
class Connectivity(object):
    """
    Defines connectivity constraints (for one variable):
        connectivityConstraintUsage:
                    (int) indicates the usage of connectivity constraints:
                        - 0: no connectivity constraint
                        - 1: set connecting paths before simulation by successively
                             binding the nodes to be connected in a random order
                        - 2: set connecting paths before simulation by successively
                             binding the nodes to be connected beginning with
                             the pair of nodes with the smallest distance and then
                             the remaining nodes in increasing order according to
                             their distance to the set of nodes already connected;
                             the distance between two nodes is defined as the length
                             (in number of nodes) of the minimal path binding the two
                             nodes in an homogeneous connected medium according to the
                             type of connectivity connectivityType
                        - 3: check connectivity pattern during simulation

        connectivityType:
                    (string) connectivity type, possible strings:
                        - 'connect_face':
                              6-neighbors connection (by face)
                        - 'connect_face_edge':
                             18-neighbors connection (by face or edge)
                        - 'connect_face_edge_corner':
                             26-neighbors connection (by face, edge or corner)

        nclass:     (int) number of classes of values
                        (unused if connectivityConstraintUsage == 0)
        classInterval:
                    (list of nclass 2-dimensional array of floats with 2 columns)
                        definition of the classes of values by intervals,
                        classInterval[i] is a (n_i, 2) array a, defining the
                        i-th class as the union of intervals:
                            [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][

        varname:    (string)
                        variable name for connected component label (should be
                        in a conditioning data set)
                        Note: label negative or zero means no connectivity constraint

        tiAsRefFlag:
                    (bool) indicates that the (first) training image is
                        used as reference for connectivity (True) or that
                        the reference image for connectivity is given by
                        refConnectivityImage (False, possible only if
                        connectivityConstraintUsage == 1 or 2)

        refConnectivityImage:
                    (Img (class), or None) reference image for connectivity
                        (used only if tiAsRefFlag is False)

        refConnectivityVarIndex:
                    (int) variable index in image refConnectivityImage for
                        the search of connected paths
                        (used only if tiAsRefFlag is False)

        deactivationDistance:
                    (float) deactivation distance (the connectivity constraint
                        is deactivated if the distance between the current
                        simulated node and the last node in its neighbors (used
                        for the search in the TI) (distance computed according
                        to the corresponding search neighborhood parameters) is
                        below the given deactivation distance),
                        used when connectivityConstraintUsage == 3

        threshold:  (float) threshold value for connectivity patterns comparison,
                        used when connectivityConstraintUsage == 3
    """

    def __init__(self,
                 connectivityConstraintUsage=0,
                 connectivityType='connect_face',
                 nclass=0,
                 classInterval=None,
                 varname='',
                 tiAsRefFlag=True,
                 refConnectivityImage=None,
                 refConnectivityVarIndex=0,
                 deactivationDistance=0.,
                 threshold=0.01):
        self.connectivityConstraintUsage = connectivityConstraintUsage
        self.connectivityType = connectivityType
        self.nclass = nclass
        if classInterval is None:
            self.classInterval = classInterval
        else:
            self.classInterval = [np.asarray(ci).reshape(-1,2) for ci in classInterval]

        self.varname = varname

        self.tiAsRefFlag = tiAsRefFlag

        if not tiAsRefFlag and refConnectivityImage is None:
            print('ERROR: (Connectivity) field "refConnectivityImage"...')
            return

        self.refConnectivityImage = refConnectivityImage
        self.refConnectivityVarIndex = refConnectivityVarIndex
        self.deactivationDistance = deactivationDistance
        self.threshold = threshold
# ============================================================================

# ============================================================================
class PyramidGeneralParameters(object):
    """
    Defines the general parameters for pyramids (all variables):
        npyramidLevel:
                    (int) number of pyramid level(s) (in addition to original
                        simulation grid), integer greater than or equal to zero,
                        if positive, pyramid is used and pyramid levels are
                        indexed from fine to coarse:
                            - index 0            : original simulation grid
                            - index npyramidLevel: coarsest level

        kx:         (1-dimensional array of ints of size npyramidLevel)
                        reduction step along x-direction for each level:
                            - kx[.] = 0: nothing is done, same dimension after
                                         reduction
                            - kx[.] = 1: same dimension after reduction
                                         (with weighted average over 3 nodes)
                            - kx[.] = 2: classical gaussian pyramid
                            - kx[.] > 2: generalized gaussian pyramid
                        (unused if npyramidLevel == 0)

        ky:         (1-dimensional array of ints of size npyramidLevel)
                        reduction step along y-direction for each level:
                            - ky[.] = 0: nothing is done, same dimension after
                                         reduction
                            - ky[.] = 1: same dimension after reduction
                                         (with weighted average over 3 nodes)
                            - ky[.] = 2: classical gaussian pyramid
                            - ky[.] > 2: generalized gaussian pyramid
                        (unused if npyramidLevel == 0)

        kz:         (1-dimensional array of ints of size npyramidLevel)
                        reduction step along z-direction for each level:
                            - kz[.] = 0: nothing is done, same dimension after
                                         reduction
                            - kz[.] = 1: same dimension after reduction
                                         (with weighted average over 3 nodes)
                            - kz[.] = 2: classical gaussian pyramid
                            - kz[.] > 2: generalized gaussian pyramid
                        (unused if npyramidLevel == 0)

        pyramidSimulationMode:
                    (string) simulation mode for pyramids, possible values:
                        - 'hierarchical':
                            (a) spreading conditioning data through the pyramid
                                by simulation at each level, from fine to coarse,
                                conditioned to the level one rank finer
                            (b) simulation at the coarsest level, then simulation
                                of each level, from coarse to fine, conditioned
                                to the level one rank coarser
                        - 'hierarchical_using_expansion':
                            (a) spreading conditioning data through the pyramid
                                by simulation at each level, from fine to coarse,
                                conditioned to the level one rank finer
                            (b) simulation at the coarsest level, then simulation
                                of each level, from coarse to fine, conditioned to
                                the gaussian expansion of the level one rank coarser
                        - 'all_level_one_by_one':
                            co-simulation of all levels, simulation done at one
                            level at a time

        factorNneighboringNode:
                    (1-dimensional array of doubles) factors for adpating the
                        maximal number of neighboring nodes,
                        - if pyramidSimulationMode == 'hierarchical' or
                             pyramidSimulationMode == 'hierarchical_using_expansion':
                            array of size 4 * npyramidLevel + 1 with entries:
                               faCond[0], faSim[0], fbCond[0], fbSim[0],
                               ...,
                               faCond[n-1], faSim[n-1], fbCond[n-1], fbSim[n-1],
                               fbSim[n]:
                            i.e. (4*n+1) positive numbers where n = npyramidLevel,
                            with the following meaning. The maximal number of
                            neighboring nodes (according to each variable)
                            is multiplied by
                                (a) faCond[j] and faSim[j] for the conditioning level (level j)
                                    and the simulated level (level j+1) resp. during step (a) above
                                (b) fbCond[j] and fbSim[j] for the conditioning level (level j+1)
                                    (expanded if pyramidSimulationMode == 'hierarchical_using_expansion')
                                    and the simulated level (level j) resp. during step (b) above
                        - if pyramidSimulationMode == all_level_one_by_one':
                            array of size npyramidLevel + 1 with entries:
                               f[0],...,f[npyramidLevel-1],f[npyramidLevel]
                            i.e. (npyramidLevel + 1) positive numbers, with the
                            following meaning. The maximal number of neighboring
                            nodes (according to each variable) is multiplied
                            by f[j] for the j-th pyramid level

        factorDistanceThreshold:
                    (1-dimensional array of floats) factors for adpating the
                        distance (acceptance) threshold (similar to factorNneighboringNode)

        factorMaxScanFraction:
                    (1-dimensional array of floats of size npyramidLevel + 1)
                        factors for adpating the maximal scan fraction:
                        the maximal scan fraction (according to each training image)
                        is multiplied by factorMaxScanFraction[j] for the j-th pyramid level
    """

    def __init__(self,
                 npyramidLevel=0,
                 nx=100, ny=100, nz=100,
                 kx=None, ky=None, kz=None,
                 pyramidSimulationMode='hierarchical_using_expansion',
                 factorNneighboringNode=None,
                 factorDistanceThreshold=None,
                 factorMaxScanFraction=None):
        """
        Init Function for the class:
            note that the parameters nx, ny, nz should be the dimension
            of the simulation grid
        """
        self.npyramidLevel = npyramidLevel

        # pyramidSimulationMode
        if pyramidSimulationMode not in ('hierarchical', 'hierarchical_using_expansion', 'all_level_one_by_one'):
            print('ERROR: (PyramidGeneralParameters) unknown pyramidSimulationMode')
            return

        self.pyramidSimulationMode = pyramidSimulationMode

        if npyramidLevel > 0:
            # kx, ky, kz
            if kx is None:
                self.kx = np.array([2 * int (nx>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kx = np.asarray(kx, dtype='int').reshape(npyramidLevel)
                except:
                    print('ERROR: (PyramidGeneralParameters) field "kx"...')
                    return

            if ky is None:
                self.ky = np.array([2 * int (ny>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.ky = np.asarray(ky, dtype='int').reshape(npyramidLevel)
                except:
                    print('ERROR: (PyramidGeneralParameters) field "kz"...')
                    return

            if kz is None:
                self.kz = np.array([2 * int (nz>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kz = np.asarray(kz, dtype='int').reshape(npyramidLevel)
                except:
                    print('ERROR: (PyramidGeneralParameters) field "kz"...')
                    return

            # factorNneighboringNode, factorDistanceThreshold
            if pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
                n = 4*npyramidLevel + 1
                if factorNneighboringNode is None:
                    factorNneighboringNode = 0.5 * np.ones(n)
                    for j in range(npyramidLevel):
                        factorNneighboringNode[4*j+3] = 0.5**min(2., npyramidLevel-j)
                        factorNneighboringNode[4*j+2] = factorNneighboringNode[4*j+3] / 3.
                    factorNneighboringNode[4*npyramidLevel] = 1.
                    self.factorNneighboringNode = factorNneighboringNode
                else:
                    try:
                        self.factorNneighboringNode = np.asarray(factorNneighboringNode, dtype=float).reshape(n)
                    except:
                        print('ERROR: (PyramidGeneralParameters) field "factorNneighboringNode"...')
                        return

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        print('ERROR: (PyramidGeneralParameters) field "factorDistanceThreshold"...')
                        return

            else: # pyramidSimulationMode == 'all_level_one_by_one'
                n = npyramidLevel + 1
                if factorNneighboringNode is None:
                    factorNneighboringNode = 1./n * np.ones(n)
                    self.factorNneighboringNode = factorNneighboringNode
                else:
                    try:
                        self.factorNneighboringNode = np.asarray(factorNneighboringNode, dtype=float).reshape(n)
                    except:
                        print('ERROR: (PyramidGeneralParameters) field "factorNneighboringNode"...')
                        return

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        print('ERROR: (PyramidGeneralParameters) field "factorDistanceThreshold"...')
                        return

            # factorMaxScanFraction
            n = npyramidLevel + 1
            if factorMaxScanFraction is None:
                factorMaxScanFraction = np.ones(n)
                d = 1.0
                for j in range(npyramidLevel):
                    d = d * np.maximum(self.kx[j], 1) * np.maximum(self.ky[j], 1) * np.maximum(self.kz[j], 1)
                    factorMaxScanFraction[j+1] = d
                self.factorMaxScanFraction = factorMaxScanFraction
            else:
                try:
                    self.factorMaxScanFraction = np.asarray(factorMaxScanFraction, dtype=float).reshape(n)
                except:
                    print('ERROR: (PyramidGeneralParameters) field "factorMaxScanFraction"...')
                    return
# ============================================================================

# ============================================================================
class PyramidParameters(object):
    """
    Defines the parameters for pyramid for one variable:
        nlevel:     (int) number of pyramid level(s) (in addition to original
                        simulation grid)

        pyramidType:
                    (string) type of pyramid, possible values:
                        - 'none':
                            no pyramid simulation
                        - 'continuous':
                            pyramid applied to continuous variable (direct)
                        - 'categorical_auto':
                            pyramid for categorical variable, pyramid for
                            indicator variable of each category except one
                            (one pyramid per indicator variable)
                        - 'categorical_custom':
                            pyramid for categorical variable, pyramid for
                            indicator variable of each class of values given
                            explicitly (one pyramid per indicator variable)
                        - 'categorical_to_continuous':
                            pyramid for categorical variable, the variable is
                            transformed to a continuous variable (according to
                            connection between adjacent nodes, the new values
                            are ordered such that close values correspond to the
                            most connected categories), then one pyramid for the
                            transformed variable is used

        nclass:     (int) number of classes of values
                        (used when pyramidType == 'categorical_custom')
        classInterval:
                    (list of nclass 2-dimensional array of floats with 2 columns)
                        definition of the classes of values by intervals,
                        classInterval[i] is a (n_i, 2) array a, defining the
                        i-th class as the union of intervals:
                            [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][
                        (used when pyramidType == 'categorical_custom')
    """

    def __init__(self,
                 nlevel=0,
                 pyramidType='none',
                 nclass=0,
                 classInterval=None):
        self.nlevel = nlevel

        if pyramidType not in ('none', 'continuous', 'categorical_auto', 'categorical_custom', 'categorical_to_continuous'):
            print('ERROR: (PyramidParameters) unknown pyramidType')
            return

        self.pyramidType = pyramidType

        self.nclass = nclass
        self.classInterval = classInterval
# ============================================================================

# ============================================================================
class DeesseInput(object):
    """
    Defines input for deesse:
        nx, ny, nz: (ints) number of simulation grid (SG) cells in each direction
        sx, sy, sz: (floats) cell size in each direction
        ox, oy, oz: (floats) origin of the SG (bottom-lower-left corner)
        nv:         (int) number of variable(s) / attribute(s), should be
                        at least 1

        varname:    (list of strings of length nv) variable names

        outputVarFlag:
                    (1-dimensional array of 'bool', of size nv)
                        flag indicating which variable is saved in output

        outputPathIndexFlag:
                    (bool) indicates if path index maps are retrieved in output
                        path index map: index in the simulation path

        outputErrorFlag:
                    (bool) indicates if error maps are retrieved in output
                        error map: error for the retained candidate

        outputTiGridNodeIndexFlag:
                    (bool) indicates if TI grid node index maps are retrieved in
                        TI grid node index map: index of the grid node of the
                        retained candidate in the TI

        outputTiIndexFlag:
                    (bool) indicates if TI index maps are retrieved in output
                        TI index map: index of the TI used (makes sense if
                        number of TIs (nTI) is greater than 1)

        outputReportFlag:
                    (bool) indicates if a report file will be written
        outputReportFileName:
                    (string) name of the report file (unused if
                        outputReportFlag is False)

        nTI:        (int) number of training image(s) (TI), should be at least 1
        simGridAsTiFlag:
                    (1-dimensional array of 'bool', of size nTI)
                        flag indicating if the simulation grid itself is used
                        as TI, for each TI

        TI:         (1-dimensional array of Img (class), of size nTI) TI(s) used
                        for the simulation
        pdfTI:      ((nTI, nz, ny, nx) array of floats) probability for TI
                        selection, pdf[i] is the "map defined on the SG" of the
                        probability to select the i-th TI, unused if nTI is less
                        than or equal to 1

        dataImage:  (1-dimensional array of Img (class), or None) data images
                        used as conditioning data (if any), each data image
                        should have the same grid dimensions as thoes of the SG
                        and its variable name(s) should be included in 'varname';
                        note that the variable names should be distinct, and each
                        data image initialize the corresponding variable in the SG
        dataPointSet:
                    (1-dimensional array of PointSet (class), or None) point sets
                        defining hard data (if any), each point set should have
                        at least 4 variables: 'X', 'Y', 'Z', the coordinates in
                        the SG and at least one variable with name in 'varname'

        mask:       ((nz, ny, nx) array of ints, or None) if given, mask values
                        over the SG: 1 for simulated node / 0 for not simulated
                        node

        homothetyUsage:
                    (int) indicates the usage of homothety (0/1/2):
                        - 0: no homothety
                        - 1: homothety without tolerance
                        - 2: homothety with tolerance
        homothetyXLocal:
                    (bool) indicates if homothety according to X axis is local
                        (True) or global (False)
                        (unused if homothetyUsage == 0)
        homothetyXRatio:
                    (nd array or None) homothety ratio according to X axis:
                        if homothetyUsage == 1:
                            if homothetyXLocal is True:
                                ((nz, ny, nx) array of floats) values on the SG
                            else:
                                (1-dimensional array of 1 float) value
                        if homothetyUsage == 2:
                            if homothetyXLocal is True:
                                ((2, nz, ny, nx) array of floats) min (homothetyXRatio[0])
                                    and max (homothetyXRatio[1]) values on the SG
                            else:
                                (1-dimensional array of 2 floats) min and max values
                        (unused if homothetyUsage == 0)
        homothetyYLocal, homothetyYRatio:
                    as homothetyXLocal and homothetyXRatio, but for the Y axis
        homothetyZLocal, homothetyZRatio:
                    as homothetyXLocal and homothetyXRatio, but for the Z axis

        rotationUsage:
                    (int) indicates the usage of rotation (0/1/2):
                        - 0: no rotation
                        - 1: rotation without tolerance
                        - 2: rotation with tolerance
        rotationAzimuthLocal:
                    (bool) indicates if azimuth angle is local (True) or
                        global (False)
                        (unused if rotationUsage == 0)
        rotationAzimuth:
                    (nd array or None) azimuth angle in degrees:
                        if rotationUsage == 1:
                            if rotationAzimuthLocal is True:
                                ((nz, ny, nx) array of floats) values on the SG
                            else:
                                (1-dimensional array of 1 float) value
                        if rotationUsage == 2:
                            if rotationAzimuthLocal is True:
                                ((2, nz, ny, nx) array of floats) min (rotationAzimuth[0])
                                    and max (rotationAzimuth[1]) values on the SG
                            else:
                                (1-dimensional array of 2 floats) min and max values
                        (unused if rotationUsage == 0)
        rotationDipLocal, rotationDip:
                    as rotationAzimuthLocal and rotationAzimuth, but for
                        the dip angle
        rotationPlungeLocal, rotationPlunge:
                    as rotationAzimuthLocal and rotationAzimuth, but for
                        the plunge angle

        expMax: (float) maximal expansion (negative to not check consistency):
                     the following is applied for each variable separetely:
                       - for variable with distance type set to 0 (see below):
                           * expMax >= 0:
                               if a conditioning data value is not in the set of training image values,
                               an error occurs
                           * expMax < 0:
                               if a conditioning data value is not in the set of training image values,
                               a warning is displayed (no error occurs)
                       - for variable with distance type not set to 0 (see below): if relative distance
                         flag is set to 1 (see below), nothing is done, else:
                           * expMax >= 0:
                               maximal accepted expansion of the range of the training image values
                               for covering the conditioning data values:
                                 - if conditioning data values are within the range of the training image values:
                                   nothing is done
                                 - if a conditioning data value is out of the range of the training image values:
                                   let
                                      new_min_ti = min ( min_cd, min_ti )
                                      new_max_ti = max ( max_cd, max_ti )
                                   with
                                      min_cd, max_cd, the min and max of the conditioning values,
                                      min_ti, max_ti, the min and max of the training imges values.
                                   If new_max_ti - new_min_ti <= (1 + expMax) * (ti_max - ti_min), then
                                   the training image values are linearly rescaled from [ti_min, ti_max] to
                                   [new_ti_min, new_ti_max], and a warning is displayed (no error occurs).
                                   Otherwise, an error occurs.
                           * expMax < 0:
                                if a conditioning data value is out of the range of the training image
                                values, a warning is displayed (no error occurs), the training image values are
                                not modified

        normalizingType:
                (string) normalizing type for non categorical variable
                    (distance type not equal to 0), possible strings:
                    'linear', 'uniform', 'normal'

        searchNeighborhoodParameters:
                (1-dimensional array of SearchNeighborhoodParameters (class) of size nv)
                    search neighborhood parameters for each variable
        nneighboringNode:
                (1-dimensional array of ints of size nv) maximal number of neighbors
                    in the search pattern, for each variable
        maxPropInequalityNode:
                (1-dimensional array of doubles of size nv) maximal proportion of nodes
                    with inequality data in the search pattern, for each variable
        neighboringNodeDensity:
                (1-dimensional array of doubles of size nv) density of neighbors
                    in the search pattern, for each variable

        rescalingMode:
                (list of strings of length nv) rescaling mode for each
                    variable, possible strings:
                    'none', 'min_max', 'mean_length'
        rescalingTargetMin:
                (1-dimensional array of doubles of size nv) target min value,
                for each variable (used for variable with rescalingMode set to
                'min_max')
        rescalingTargetMax:
                (1-dimensional array of doubles of size nv) target max value,
                for each variable (used for variable with rescalingMode set to
                'min_max')
        rescalingTargetMean:
                (1-dimensional array of doubles of size nv) target mean value,
                for each variable (used for variable with rescalingMode set to
                'mean_length')
        rescalingTargetLength:
                (1-dimensional array of doubles of size nv) target length value,
                for each variable (used for variable with rescalingMode set to
                'mean_length')

        relativeDistanceFlag:
                (1-dimensional array of 'bool', of size nv)
                    flag for each variable indicating if relative distance
                    is used (True) or not (False)
        distanceType:
                (List (or 1-dimensional array) of ints or strings of size nv)
                    distance type (between pattern) for each variable; possible value:
                        - 0 or 'categorical' : non-matching nodes (default if None)
                        - 1 or 'continuous'  : L-1 distance
                        - 2 : L-2 distance
                        - 3 : L-p distance, requires the real positive parameter p
                        - 4 : L-infinity
        powerLpDistance
                (1-dimensional array of doubles of size nv) p parameter for L-p
                    distance, for each variable (unused for variable not using
                    L-p distance)
        powerLpDistanceInv
                (1-dimensional array of doubles of size nv) inverse of p parameter
                    for L-p distance, for each variable (unused for variable not
                    using L-p distance)

        conditioningWeightFactor:
                (1-dimensional array of floats of size nv) weight factor for
                conditioning data, for each variable

        simType:(string) simulation type, possible strings:
                    - 'sim_one_by_one': successive simulation of one variable at
                        one node in the simulation grid (4D path)
                    - 'sim_variable_vector': successive simulation of all
                        variable(s) at one node in the simulation grid (3D path)
        simPathType:
                (string) simulation path type: possible strings:
                    - 'random': random path
                    - 'random_hd_distance_pdf': random path set according to distance
                        to conditioning nodes based on pdf,
                        required field 'simPathStrength', see below
                    - 'random_hd_distance_sort': random path set according to distance
                        to conditioning nodes based on sort (with a random noise
                        contribution),
                        required field 'simPathStrength', see below
                    - 'random_hd_distance_sum_pdf': random path set according to sum
                        of distance to conditioning nodes based on pdf,
                        required fields 'simPathPower' and 'simPathStrength', see below
                    - 'random_hd_distance_sum_sort': random path set according to sum
                        of distance to conditioning nodes based on sort (with a random
                        noise contribution),
                        required fields 'simPathPower' and 'simPathStrength', see below
                    - 'unilateral': unilateral path or stratified random path,
                        required field 'simPathUnilateralOrder', see below
        simPathStrength:
                (double) strength in [0,1] attached to distance if simPathType is
                'random_hd_distance_pdf' or 'random_hd_distance_sort' or
                'random_hd_distance_sum_pdf' or 'random_hd_distance_sum_sort'
                 (unused otherwise)
        simPathPower:
                (double) power (>0) to which the distance to each conditioning node
                are elevated, if simPathType is
                'random_hd_distance_sum_pdf' or 'random_hd_distance_sum_sort'
                 (unused otherwise)
        simPathUnilateralOrder:
                (1-dimesional array of ints), used when simPathType == 'unilateral'
                    - if simType == 'sim_one_by_one': simPathUnilateralOrder is
                        of length 4, example: [0, -2, 1, 0] means that the path will
                        visit all nodes: randomly in xv-sections, with increasing
                        z-coordinate, and then decreasing y-coordinate
                    - if simType == 'sim_variable_vector': simPathUnilateralOrder is
                        of length 3, example: [-1, 0, 2] means that the path will
                        visit all nodes: randomly in y-sections, with decreasing
                        x-coordinate, and then increasing z-coordinate

        distanceThreshold:
                (1-dimensional array of floats of size nv) distance (acceptance)
                    threshold for each variable

        softProbability:
                (1-dimensional array of SoftProbability (class) of size nv)
                    probability constraints parameters for each variable

        connectivity:
                (1-dimensional array of Connectivity (class) of size nv)
                    connectivity constraints parameters for each variable

        blockData:
                (1-dimensional array of BlockData (class) of size nv)
                    block data parameters for each variable

        maxScanFraction:
                (1-dimensional array of doubles of size nTI)
                    maximal scan fraction of each TI

        pyramidGeneralParameters:
                (PyramidGeneralParameters (class))
                    defines the general pyramid parameters

        pyramidParameters:
                (1-dimensional array of PyramidParameters (class) of size nv)
                    pyramid parameters for each variable

        tolerance:
                (float) tolerance on the (acceptance) threshold value for flagging
                    nodes (for post-processing)

        npostProcessingPathMax:
                (int) maximal number of post-processing path(s)
                    (0 for no post-processing)

        postProcessingNneighboringNode:
                (1-dimensional array of ints of size nv) maximal number of neighbors
                    in the search pattern, for each variable (for all post-processing
                    paths)

        postProcessingNeighboringNodeDensity:
                (1-dimensional array of doubles of size nv) density of neighbors
                    in the search pattern, for each variable (for all post-processing
                    paths)

        postProcessingDistanceThreshold:
                (1-dimensional array of floats of size nv) distance (acceptance)
                    threshold for each variable (for all post-processing paths)

        postProcessingMaxScanFraction:
                (1-dimensional array of doubles of size nTI) maximal scan fraction
                    of each TI (for all post-processing paths)

        postProcessingTolerance:
                (float) tolerance on the (acceptance) threshold value for flagging
                    nodes (for post-processing) (for all post-processing paths)

        seed:   (int) initial seed
        seedIncrement:
                (int) increment seed

        nrealization:
                (int) number of realization(s)

        name:   (string) name of the set of parameters
    """

    def __init__(self,
                 nx=0,   ny=0,   nz=0,
                 sx=1.0, sy=1.0, sz=1.0,
                 ox=0.0, oy=0.0, oz=0.0,
                 nv=0, varname=None, outputVarFlag=None,
                 outputPathIndexFlag=False, #outputPathIndexFileName=None,
                 outputErrorFlag=False, #outputErrorFileName=None,
                 outputTiGridNodeIndexFlag=False, #outputTiGridNodeIndexFileName=None,
                 outputTiIndexFlag=False, #outputTiIndexFileName=None,
                 outputReportFlag=False, outputReportFileName='ds.log',
                 nTI=0, simGridAsTiFlag=None, TI=None, pdfTI=None,
                 dataImage=None, dataPointSet=None,
                 mask=None,
                 homothetyUsage=0,
                 homothetyXLocal=False, homothetyXRatio=None,
                 homothetyYLocal=False, homothetyYRatio=None,
                 homothetyZLocal=False, homothetyZRatio=None,
                 rotationUsage=0,
                 rotationAzimuthLocal=False, rotationAzimuth=None,
                 rotationDipLocal=False,     rotationDip=None,
                 rotationPlungeLocal=False,  rotationPlunge=None,
                 expMax=0.05,
                 normalizingType='linear',
                 searchNeighborhoodParameters=None,
                 nneighboringNode=None,
                 maxPropInequalityNode=None, neighboringNodeDensity=None,
                 rescalingMode=None,
                 rescalingTargetMin=None, rescalingTargetMax=None,
                 rescalingTargetMean=None, rescalingTargetLength=None,
                 relativeDistanceFlag=None,
                 distanceType=None,
                 powerLpDistance=None,
                 conditioningWeightFactor=None,
                 simType='sim_one_by_one',
                 simPathType='random',
                 simPathStrength=0.5,
                 simPathPower=2.0,
                 simPathUnilateralOrder=None,
                 distanceThreshold=None,
                 softProbability=None,
                 connectivity=None,
                 blockData=None,
                 maxScanFraction=None,
                 tolerance=0.0,
                 npostProcessingPathMax=0,
                 postProcessingNneighboringNode=None,
                 postProcessingNeighboringNodeDensity=None,
                 postProcessingDistanceThreshold=None,
                 postProcessingMaxScanFraction=None,
                 postProcessingTolerance=0.,
                 pyramidGeneralParameters=None,
                 pyramidParameters=None,
                 seed=1234,
                 seedIncrement=1,
                 nrealization=1):
        self.ok = False # flag to "validate" the class
        self.consoleAppFlag = False
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.nv = nv
        if varname is None:
            self.varname = ["V{:d}".format(i) for i in range(nv)]
        else:
            try:
                self.varname = list(np.asarray(varname).reshape(nv))
            except:
                print('ERROR: (DeesseInput) field "varname"...')
                return

        if outputVarFlag is None:
            self.outputVarFlag = np.array([True for i in range(nv)], dtype='bool')
        else:
            try:
                self.outputVarFlag = np.asarray(outputVarFlag, dtype='bool').reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "outputVarFlag"...')
                return

        self.outputPathIndexFlag = outputPathIndexFlag
        # self.outputPathIndexFileName = None # no output file!

        self.outputErrorFlag = outputErrorFlag
        # self.outputErrorFileName = None # no output file!

        self.outputTiGridNodeIndexFlag = outputTiGridNodeIndexFlag
        # self.outputTiGridNodeIndexFileName = None # no output file!

        self.outputTiIndexFlag = outputTiIndexFlag
        # self.outputTiIndexFileName = None # no output file!

        self.outputReportFlag = outputReportFlag
        self.outputReportFileName = outputReportFileName

        dim = int(nx>1) + int(ny>1) + int(nz>1)

        self.nTI = nTI

        if simGridAsTiFlag is None:
            self.simGridAsTiFlag = np.array([False for i in range(nTI)], dtype='bool')
        else:
            try:
                self.simGridAsTiFlag = np.asarray(simGridAsTiFlag, dtype='bool').reshape(nTI)
            except:
                print('ERROR: (DeesseInput) field "simGridAsTiFlag"...')
                return

        if TI is None:
            self.TI = np.array([None for i in range(nTI)], dtype=object)
        else:
            try:
                self.TI = np.asarray(TI).reshape(nTI)
            except:
                print('ERROR: (DeesseInput) field "TI"...')
                return

        for f, t in zip(self.simGridAsTiFlag, self.TI):
            if not f and t is None:
                print ('ERROR: (DeesseInput) invalid "TI / simGridAsTiFlag"...')
                return

        if self.nTI <= 1:
            self.pdfTI = None
        else:
            if pdfTI is None:
                p = 1./self.nTI
                self.pdfTI = np.repeat(p, self.nTI*nx*ny*nz).reshape(self.nTI, nz, ny, nx)
            else:
                try:
                    self.pdfTI = np.asarray(pdfTI, dtype=float).reshape(self.nTI, nz, ny, nx)
                except:
                    print('ERROR: (DeesseInput) field "pdfTI"...')
                    return

        if dataImage is None:
            self.dataImage = None
        else:
            self.dataImage = np.asarray(dataImage).reshape(-1)

        if dataPointSet is None:
            self.dataPointSet = None
        else:
            self.dataPointSet = np.asarray(dataPointSet).reshape(-1)

        if mask is None:
            self.mask = None
        else:
            try:
                self.mask = np.asarray(mask).reshape(nz, ny, nx)
            except:
                print('ERROR: (DeesseInput) field "mask"...')
                return

        if homothetyUsage == 1:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyXRatio"...')
                        return
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyXRatio"...')
                        return

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyYRatio"...')
                        return
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyYRatio"...')
                        return

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyZRatio"...')
                        return
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyZRatio"...')
                        return

        elif homothetyUsage == 2:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyXRatio"...')
                        return
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyXRatio"...')
                        return

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyYRatio"...')
                        return
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyYRatio"...')
                        return

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyZRatio"...')
                        return
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "homothetyZRatio"...')
                        return

        elif homothetyUsage != 0:
            print ('ERROR: (DeesseInput) invalid homothetyUsage')
            return

        self.homothetyUsage = homothetyUsage
        self.homothetyXLocal = homothetyXLocal
        self.homothetyYLocal = homothetyYLocal
        self.homothetyZLocal = homothetyZLocal

        if rotationUsage == 1:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationAzimuth"...')
                        return
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "rotationAzimuth"...')
                        return

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationDip"...')
                        return
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "rotationDip"...')
                        return

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationPlunge"...')
                        return
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(1)
                    except:
                        print('ERROR: (DeesseInput) field "rotationPlunge"...')
                        return

        elif rotationUsage == 2:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationAzimuth"...')
                        return
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0., 0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "rotationAzimuth"...')
                        return

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationDip"...')
                        return
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0., 0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "rotationDip"...')
                        return

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print('ERROR: (DeesseInput) field "rotationPlunge"...')
                        return
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0., 0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2)
                    except:
                        print('ERROR: (DeesseInput) field "rotationPlunge"...')
                        return

        elif rotationUsage != 0:
            print ('ERROR: (DeesseInput) invalid rotationUsage')
            return

        self.rotationUsage = rotationUsage
        self.rotationAzimuthLocal = rotationAzimuthLocal
        self.rotationDipLocal = rotationDipLocal
        self.rotationPlungeLocal = rotationPlungeLocal

        self.expMax = expMax

        # if normalizingType not in ('linear', 'uniform', 'normal'):
        #     print ('ERRROR: unknown normalizingType')
        #     return

        self.normalizingType = normalizingType

        if searchNeighborhoodParameters is None:
            self.searchNeighborhoodParameters = np.array([SearchNeighborhoodParameters() for i in range(nv)])
        else:
            try:
                self.searchNeighborhoodParameters = np.asarray(searchNeighborhoodParameters).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "searchNeighborhoodParameters"...')
                return

        if nneighboringNode is None:
            if dim == 3: # 3D simulation
                n = 36
            else:
                n = 24

            if nv > 1:
                n = int(np.ceil(n/nv))

            self.nneighboringNode = np.array([n for i in range(nv)])
        else:
            try:
                self.nneighboringNode = np.asarray(nneighboringNode).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "nneighboringNode"...')
                return

        if maxPropInequalityNode is None:
            self.maxPropInequalityNode = np.array([0.25 for i in range(nv)])
        else:
            try:
                self.maxPropInequalityNode = np.asarray(maxPropInequalityNode).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "maxPropInequalityNode"...')
                return

        if neighboringNodeDensity is None:
            self.neighboringNodeDensity = np.array([1. for i in range(nv)])
        else:
            try:
                self.neighboringNodeDensity = np.asarray(neighboringNodeDensity, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "neighboringNodeDensity"...')
                return

        if rescalingMode is None:
            self.rescalingMode = ['none' for i in range(nv)]
        else:
            try:
                self.rescalingMode = list(np.asarray(rescalingMode).reshape(nv))
            except:
                print('ERROR: (DeesseInput) field "rescalingMode"...')
                return

        if rescalingTargetMin is None:
            self.rescalingTargetMin = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMin = np.asarray(rescalingTargetMin, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "rescalingTargetMin"...')
                return

        if rescalingTargetMax is None:
            self.rescalingTargetMax = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMax = np.asarray(rescalingTargetMax, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "rescalingTargetMax"...')
                return

        if rescalingTargetMean is None:
            self.rescalingTargetMean = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMean = np.asarray(rescalingTargetMean, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "rescalingTargetMean"...')
                return

        if rescalingTargetLength is None:
            self.rescalingTargetLength = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetLength = np.asarray(rescalingTargetLength, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "rescalingTargetLength"...')
                return

        if relativeDistanceFlag is None:
            self.relativeDistanceFlag = np.array([False for i in range(nv)])
        else:
            try:
                self.relativeDistanceFlag = np.asarray(relativeDistanceFlag, dtype='bool').reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "relativeDistanceFlag"...')
                return

        if powerLpDistance is None:
            self.powerLpDistance = np.array([1. for i in range(nv)])
        else:
            try:
                self.powerLpDistance = np.asarray(powerLpDistance, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "powerLpDistance"...')
                return

        self.powerLpDistanceInv = 1./self.powerLpDistance

        if distanceType is None:
            self.distanceType = np.array([0 for i in range(nv)])
        else:
            try:
                if isinstance(distanceType, str) or isinstance(distanceType, int):
                    self.distanceType = [distanceType]
                else:
                    self.distanceType = list(distanceType)
                for i in range(len(self.distanceType)):
                    if isinstance(self.distanceType[i], str):
                        if self.distanceType[i] == 'categorical':
                            self.distanceType[i] = 0
                        elif self.distanceType[i] == 'continuous':
                            self.distanceType[i] = 1
                        else:
                            print('ERROR: (DeesseInput) field "distanceType"...')
                            return
                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "distanceType"...')
                return

        if conditioningWeightFactor is None:
            self.conditioningWeightFactor = np.array([1. for i in range(nv)])
        else:
            try:
                self.conditioningWeightFactor = np.asarray(conditioningWeightFactor, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "conditioningWeightFactor"...')
                return

        if simType not in ('sim_one_by_one', 'sim_variable_vector'):
            print ('ERRROR: unknown simType')
            return

        self.simType = simType

        if simPathType not in ('random',
            'random_hd_distance_pdf', 'random_hd_distance_sort',
            'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort',
            'unilateral'):
            print ('ERRROR: unknown simPathType')
            return

        self.simPathType = simPathType

        self.simPathPower = simPathPower
        self.simPathStrength = simPathStrength

        if simPathType == 'unilateral':
            if simType == 'sim_one_by_one':
                length = 4
            else: # simType == 'sim_variable_vector':
                length = 3

            if simPathUnilateralOrder is None:
                self.simPathUnilateralOrder = np.array([i+1 for i in range(length)])
            else:
                try:
                    self.simPathUnilateralOrder = np.asarray(simPathUnilateralOrder).reshape(length)
                except:
                    print('ERROR: (DeesseInput) field "simPathUnilateralOrder"...')
                    return
        else:
            self.simPathUnilateralOrder = None

        if distanceThreshold is None:
            self.distanceThreshold = np.array([0.05 for i in range(nv)])
        else:
            try:
                self.distanceThreshold = np.asarray(distanceThreshold, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "distanceThreshold"...')
                return

        if softProbability is None:
            self.softProbability = np.array([SoftProbability(probabilityConstraintUsage=0) for i in range(nv)])
        else:
            try:
                self.softProbability = np.asarray(softProbability).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "softProbability"...')
                return

        if connectivity is None:
            self.connectivity = np.array([Connectivity(connectivityConstraintUsage=0) for i in range(nv)])
        else:
            try:
                self.connectivity = np.asarray(connectivity).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "connectivity"...')
                return

        if blockData is None:
            self.blockData = np.array([BlockData(blockDataUsage=0) for i in range(nv)])
        else:
            try:
                self.blockData = np.asarray(blockData).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "blockData"...')
                return

        if maxScanFraction is None:
            if dim == 3: # 3D simulation
                nf = 10000
            else:
                nf = 5000

            self.maxScanFraction = np.array([min(max(nf/self.TI[i].nxyz(), deesse.MPDS_MIN_MAXSCANFRACTION), deesse.MPDS_MAX_MAXSCANFRACTION) for i in range(self.nTI)])
        else:
            try:
                self.maxScanFraction = np.asarray(maxScanFraction).reshape(self.nTI)
            except:
                print('ERROR: (DeesseInput) field "maxScanFraction"...')
                return

        if pyramidGeneralParameters is None:
            self.pyramidGeneralParameters = PyramidGeneralParameters(nx=nx, ny=ny, nz=nz)
        else:
            self.pyramidGeneralParameters = pyramidGeneralParameters

        if pyramidParameters is None:
            self.pyramidParameters = np.array([PyramidParameters() for i in range(nv)])
        else:
            try:
                self.pyramidParameters = np.asarray(pyramidParameters).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "pyramidParameters"...')
                return

        self.tolerance = tolerance
        self.npostProcessingPathMax = npostProcessingPathMax

        if postProcessingNneighboringNode is None:
            if dim <= 1:
                self.postProcessingNneighboringNode = np.array([deesse.MPDS_POST_PROCESSING_NNEIGHBORINGNODE_DEFAULT_1D for i in range(nv)])
            elif dim == 2:
                self.postProcessingNneighboringNode = np.array([deesse.MPDS_POST_PROCESSING_NNEIGHBORINGNODE_DEFAULT_2D for i in range(nv)])
            else:
                self.postProcessingNneighboringNode = np.array([deesse.MPDS_POST_PROCESSING_NNEIGHBORINGNODE_DEFAULT_3D for i in range(nv)])
        else:
            try:
                self.postProcessingNneighboringNode = np.asarray(postProcessingNneighboringNode).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "postProcessingNneighboringNode"...')
                return

        if postProcessingNeighboringNodeDensity is None:
            if dim <= 1:
                self.postProcessingNeighboringNodeDensity = np.array([deesse.MPDS_POST_PROCESSING_NEIGHBORINGNODE_DENSITY_DEFAULT_1D for i in range(nv)], dtype=float)
            elif dim == 2:
                self.postProcessingNeighboringNodeDensity = np.array([deesse.MPDS_POST_PROCESSING_NEIGHBORINGNODE_DENSITY_DEFAULT_2D for i in range(nv)], dtype=float)
            else:
                self.postProcessingNeighboringNodeDensity = np.array([deesse.MPDS_POST_PROCESSING_NEIGHBORINGNODE_DENSITY_DEFAULT_3D for i in range(nv)], dtype=float)
        else:
            try:
                self.postProcessingNeighboringNodeDensity = np.asarray(postProcessingNeighboringNodeDensity, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "postProcessingNeighboringNodeDensity"...')
                return

        if postProcessingDistanceThreshold is None:
            self.postProcessingDistanceThreshold = np.zeros(nv)
            for i in range(nv):
                if self.distanceType[i] == 0:
                    self.postProcessingDistanceThreshold[i] = deesse.MPDS_POST_PROCESSING_DISTANCE_THRESHOLD_DEFAULT_DISTANCETYPE_0
                elif self.distanceType[i] == 1:
                    self.postProcessingDistanceThreshold[i] = deesse.MPDS_POST_PROCESSING_DISTANCE_THRESHOLD_DEFAULT_DISTANCETYPE_1
                elif self.distanceType[i] == 2:
                    self.postProcessingDistanceThreshold[i] = deesse.MPDS_POST_PROCESSING_DISTANCE_THRESHOLD_DEFAULT_DISTANCETYPE_2
                elif self.distanceType[i] == 3:
                    self.postProcessingDistanceThreshold[i] = deesse.MPDS_POST_PROCESSING_DISTANCE_THRESHOLD_DEFAULT_DISTANCETYPE_3
                elif self.distanceType[i] == 4:
                    self.postProcessingDistanceThreshold[i] = deesse.MPDS_POST_PROCESSING_DISTANCE_THRESHOLD_DEFAULT_DISTANCETYPE_4
        else:
            try:
                self.postProcessingDistanceThreshold = np.asarray(postProcessingDistanceThreshold, dtype=float).reshape(nv)
            except:
                print('ERROR: (DeesseInput) field "postProcessingDistanceThreshold"...')
                return

        if postProcessingMaxScanFraction is None:
            self.postProcessingMaxScanFraction = np.array([min(deesse.MPDS_POST_PROCESSING_MAX_SCAN_FRACTION_DEFAULT, self.maxScanFraction[i]) for i in range(nTI)], dtype=float)

        else:
            try:
                self.postProcessingMaxScanFraction = np.asarray(postProcessingMaxScanFraction, dtype=float).reshape(nTI)
            except:
                print('ERROR: (DeesseInput) field "postProcessingMaxScanFraction"...')
                return

        self.postProcessingTolerance = postProcessingTolerance

        self.seed = seed
        self.seedIncrement = seedIncrement
        self.nrealization = nrealization
        self.ok = True # flag to "validate" the class
# ============================================================================

# ----------------------------------------------------------------------------
def img_py2C(im_py):
    """
    Converts an image from python to C.

    :param im_py:   (Img class) image (python class)
    :return im_c:   (MPDS_IMAGE *) image converted (C struct)
    """

    im_c = deesse.malloc_MPDS_IMAGE()
    deesse.MPDSInitImage(im_c)

    err = deesse.MPDSMallocImage(im_c, im_py.nxyz(), im_py.nv)
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
        deesse.mpds_set_varname(im_c.varName, i, im_py.varname[i])

    v = im_py.val.reshape(-1)
    np.putmask(v, np.isnan(v), deesse.MPDS_MISSING_VALUE)
    deesse.mpds_set_real_vector_from_array(im_c.var, 0, v)
    np.putmask(v, v == deesse.MPDS_MISSING_VALUE, np.nan) # replace missing_value by np.nan (restore) (v is not a copy...)

    return im_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def img_C2py(im_c):
    """
    Converts an image from C to python.

    :param im_c:     (MPDS_IMAGE *) image (C struct)
    :return im_py:   (Img class) image converted (python class)
    """

    nxyz = im_c.grid.nx * im_c.grid.ny * im_c.grid.nz
    nxyzv = nxyz * im_c.nvar

    varname = [deesse.mpds_get_varname(im_c.varName, i) for i in range(im_c.nvar)]

    v = np.zeros(nxyzv)
    deesse.mpds_get_array_from_real_vector(im_c.var, 0, v)

    im_py = Img(nx=im_c.grid.nx, ny=im_c.grid.ny, nz=im_c.grid.nz,
                sx=im_c.grid.sx, sy=im_c.grid.sy, sz=im_c.grid.sz,
                ox=im_c.grid.ox, oy=im_c.grid.oy, oz=im_c.grid.oz,
                nv=im_c.nvar, val=v, varname=varname)

    np.putmask(im_py.val, im_py.val == deesse.MPDS_MISSING_VALUE, np.nan)

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

    ps_c = deesse.malloc_MPDS_POINTSET()
    deesse.MPDSInitPointSet(ps_c)

    err = deesse.MPDSMallocPointSet(ps_c, ps_py.npt, nvar)
    if err:
        print ('ERROR: can not convert point set from python to C')
        return

    ps_c.npoint = ps_py.npt
    ps_c.nvar = nvar

    for i in range(nvar):
        deesse.mpds_set_varname(ps_c.varName, i, ps_py.varname[i+3])

    deesse.mpds_set_real_vector_from_array(ps_c.x, 0, ps_py.val[0].reshape(-1))
    deesse.mpds_set_real_vector_from_array(ps_c.y, 0, ps_py.val[1].reshape(-1))
    deesse.mpds_set_real_vector_from_array(ps_c.z, 0, ps_py.val[2].reshape(-1))

    v = ps_py.val[3:].reshape(-1)
    np.putmask(v, np.isnan(v), deesse.MPDS_MISSING_VALUE)
    deesse.mpds_set_real_vector_from_array(ps_c.var, 0, v)
    np.putmask(v, v == deesse.MPDS_MISSING_VALUE, np.nan)  # replace missing_value by np.nan (restore) (v is not a copy...)

    return ps_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def ps_C2py(ps_c):
    """
    Converts a point set from C to python.

    :param ps_c:    (MPDS_POINTSET *) point set (C struct)
    :return ps_py:  (PointSet class) point set converted (python class)
    """

    varname = ['X', 'Y', 'Z'] + [deesse.mpds_get_varname(ps_c.varName, i) for i in range(ps_c.nvar)]

    v = np.zeros(ps_c.npoint*ps_c.nvar)
    deesse.mpds_get_array_from_real_vector(ps_c.var, 0, v)

    coord = np.zeros(ps_c.npoint)
    deesse.mpds_get_array_from_real_vector(ps_c.z, 0, coord)
    v = np.hstack(coord,v)
    deesse.mpds_get_array_from_real_vector(ps_c.y, 0, coord)
    v = np.hstack(coord,v)
    deesse.mpds_get_array_from_real_vector(ps_c.x, 0, coord)
    v = np.hstack(coord,v)

    ps_py = PointSet(npt=ps_c.npoint,
                     nv=ps_c.nvar+3, val=v, varname=varname)

    np.putmask(ps_py.val, ps_py.val == deesse.MPDS_MISSING_VALUE, np.nan)

    return ps_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def classInterval2classOfValues(classInterval):
    """
    Converts classInterval (python) to classOfValues (C).

    :param classInterval:
                    (list of nclass 2-dimensional array of floats with 2 columns)
                        definition of the classes of values by intervals,
                        classInterval[i] is a (n_i, 2) array a, defining the
                        i-th class as the union of intervals:
                            [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][
    :return cv:     (MPDS_CLASSOFVALUES *) corresponding structure in C
    """

    cv = deesse.malloc_MPDS_CLASSOFVALUES()
    deesse.MPDSInitClassOfValues(cv)

    n = len(classInterval)
    cv.nclass = n
    cv.ninterval = deesse.new_int_array(n)
    cv.intervalInf = deesse.new_realp_array(n)
    cv.intervalSup = deesse.new_realp_array(n)
    for j, ci in enumerate(classInterval):
        nint = len(ci) # = ci.shape[0]
        deesse.int_array_setitem(cv.ninterval, j, nint)
        intInf_c = deesse.new_real_array(nint)
        deesse.mpds_set_real_vector_from_array(intInf_c, 0, ci[:,0])
        deesse.realp_array_setitem(cv.intervalInf, j, intInf_c)
        intSup_c = deesse.new_real_array(nint)
        deesse.mpds_set_real_vector_from_array(intSup_c, 0, ci[:,1])
        deesse.realp_array_setitem(cv.intervalSup, j, intSup_c)

    return cv
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_input_py2C(deesse_input):
    """
    Converts deesse input from python to C.

    :param deesse_input: (DeesseInput class) deesse input - python
    :return:             (MPDS_SIMINPUT) deesse input - C
    """

    nxy = deesse_input.nx * deesse_input.ny
    nxyz = nxy * deesse_input.nz

    # Allocate mpds_siminput
    mpds_siminput = deesse.malloc_MPDS_SIMINPUT()

    # Init mpds_siminput
    deesse.MPDSInitSimInput(mpds_siminput)

    # mpds_siminput.consoleAppFlag
    if deesse_input.consoleAppFlag:
        mpds_siminput.consoleAppFlag = deesse.TRUE
    else:
        mpds_siminput.consoleAppFlag = deesse.FALSE

    # mpds_siminput.simImage
    # ... set initial image im (for simulation)
    im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
             sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
             ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
             nv=deesse_input.nv, val=deesse.MPDS_MISSING_VALUE,
             varname=deesse_input.varname)

    # # ... integrate data image to im
    # if deesse_input.dataImage is not None:
    #     for i, dataIm in enumerate(deesse_input.dataImage):
    #         if not img.isImageDimensionEqual(im, dataIm):
    #             print ('ERROR: invalid data image dimension')
    #             return
    #
    #         for j in range(dataIm.nv):
    #             vname = dataIm.varname[j]
    #             tmp = [vname == n for n in deesse_input.varname]
    #             if np.sum(tmp) != 1:
    #                 print('ERROR: variable name in data image does not match one variable name in SG')
    #                 return
    #
    #             iv = np.where(tmp)[0][0]
    #             im.set_var(val=dataIm.val[j,...], ind=iv)

    # ... convert im from python to C
    mpds_siminput.simImage = img_py2C(im)

    # mpds_siminput.nvar
    mpds_siminput.nvar = int(deesse_input.nv)

    # mpds_siminput.outputVarFlag
    deesse.mpds_set_outputVarFlag(mpds_siminput, np.array([int(i) for i in deesse_input.outputVarFlag], dtype='intc'))

    # mpds_siminput.formatStringVar: not used

    # mpds_siminput.outputSimJob
    mpds_siminput.outputSimJob = deesse.OUTPUT_SIM_NO_FILE

    # mpds_siminput.outputSimImageFileName: not used (NULL: no output file!)

    # mpds_siminput.outputPathIndexFlag
    if deesse_input.outputPathIndexFlag:
        mpds_siminput.outputPathIndexFlag = deesse.TRUE
    else:
        mpds_siminput.outputPathIndexFlag = deesse.FALSE

    # mpds_siminput.outputPathIndexFileName: not used (NULL: no output file!)

    # mpds_siminput.outputErrorFlag
    if deesse_input.outputErrorFlag:
        mpds_siminput.outputErrorFlag = deesse.TRUE
    else:
        mpds_siminput.outputErrorFlag = deesse.FALSE

    # mpds_siminput.outputErrorFileName: not used (NULL: no output file!)

    # mpds_siminput.outputTiGridNodeIndexFlag
    if deesse_input.outputTiGridNodeIndexFlag:
        mpds_siminput.outputTiGridNodeIndexFlag = deesse.TRUE
    else:
        mpds_siminput.outputTiGridNodeIndexFlag = deesse.FALSE

    # mpds_siminput.outputTiGridNodeIndexFileName: not used (NULL: no output file!)

    # mpds_siminput.outputTiIndexFlag
    if deesse_input.outputTiIndexFlag:
        mpds_siminput.outputTiIndexFlag = deesse.TRUE
    else:
        mpds_siminput.outputTiIndexFlag = deesse.FALSE

    # mpds_siminput.outputTiIndexFileName: not used (NULL: no output file!)

    # mpds_siminput.outputReportFlag
    if deesse_input.outputReportFlag:
        mpds_siminput.outputReportFlag = deesse.TRUE
        deesse.mpds_set_outputReportFileName(mpds_siminput, deesse_input.outputReportFileName)
    else:
        mpds_siminput.outputReportFlag = deesse.FALSE

    # mpds_siminput.ntrainImage
    mpds_siminput.ntrainImage = deesse_input.nTI

    # mpds_siminput.simGridAsTiFlag
    deesse.mpds_set_simGridAsTiFlag(mpds_siminput, np.array([int(i) for i in deesse_input.simGridAsTiFlag], dtype='intc'))

    # mpds_siminput.trainImage
    mpds_siminput.trainImage = deesse.new_MPDS_IMAGE_array(deesse_input.nTI)
    for i, ti in enumerate(deesse_input.TI):
        if ti is not None:
            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.trainImage, i, img_py2C(ti))

    # mpds_siminput.pdfTrainImage
    if deesse_input.nTI > 1:
        im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                 sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                 ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                 nv=deesse_input.nTI, val=deesse_input.pdfTI)
        mpds_siminput.pdfTrainImage = img_py2C(im)

    # mpds_siminput.ndataImage and mpds_siminput.dataImage
    if deesse_input.dataImage is None:
        mpds_siminput.ndataImage = 0
    else:
        n = len(deesse_input.dataImage)
        mpds_siminput.ndataImage = n
        mpds_siminput.dataImage = deesse.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(deesse_input.dataImage):
            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImage, i, img_py2C(dataIm))

    # mpds_siminput.ndataPointSet and mpds_siminput.dataPointSet
    if deesse_input.dataPointSet is None:
        mpds_siminput.ndataPointSet = 0
    else:
        n = len(deesse_input.dataPointSet)
        mpds_siminput.ndataPointSet = n
        mpds_siminput.dataPointSet = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesse_input.dataPointSet):
            deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSet, i, ps_py2C(dataPS))

    # mpds_siminput.maskImageFlag and mpds_siminput.maskImage
    if deesse_input.mask is None:
        mpds_siminput.maskImageFlag = deesse.FALSE
    else:
        mpds_siminput.maskImageFlag = deesse.TRUE
        im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                 sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                 ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                 nv=1, val=deesse_input.mask)
        mpds_siminput.maskImage = img_py2C(im)

    # Homothety:
    #   mpds_siminput.homothetyUsage
    #   mpds_siminput.homothety[XYZ]RatioImageFlag
    #   mpds_siminput.homothety[XYZ]RatioImage
    #   mpds_siminput.homothety[XYZ]RatioValue
    mpds_siminput.homothetyUsage = deesse_input.homothetyUsage
    if deesse_input.homothetyUsage == 1:
        if deesse_input.homothetyXLocal:
            mpds_siminput.homothetyXRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyXRatio)
            mpds_siminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyXRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyXRatioValue, 0,
                np.asarray(deesse_input.homothetyXRatio).reshape(1))

        if deesse_input.homothetyYLocal:
            mpds_siminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyYRatio)
            mpds_siminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyYRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyYRatioValue, 0,
                np.asarray(deesse_input.homothetyYRatio).reshape(1))

        if deesse_input.homothetyZLocal:
            mpds_siminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyZRatio)
            mpds_siminput.homothetyZRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyZRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyZRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyZRatioValue, 0,
                np.asarray(deesse_input.homothetyZRatio).reshape(1))

    elif deesse_input.homothetyUsage == 2:
        if deesse_input.homothetyXLocal:
            mpds_siminput.homothetyXRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyXRatio)
            mpds_siminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyXRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyXRatioValue, 0,
                np.asarray(deesse_input.homothetyXRatio).reshape(2))

        if deesse_input.homothetyYLocal:
            mpds_siminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyYRatio)
            mpds_siminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyYRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyYRatioValue, 0,
                np.asarray(deesse_input.homothetyYRatio).reshape(2))

        if deesse_input.homothetyZLocal:
            mpds_siminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyZRatio)
            mpds_siminput.homothetyZRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyZRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyZRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyZRatioValue, 0,
                np.asarray(deesse_input.homothetyZRatio).reshape(2))

    # Rotation:
    #   mpds_siminput.rotationUsage
    #   mpds_siminput.rotation[Azimuth|Dip|Plunge]ImageFlag
    #   mpds_siminput.rotation[Azimuth|Dip|Plunge]Image
    #   mpds_siminput.rotation[Azimuth|Dip|Plunge]Value
    mpds_siminput.rotationUsage = deesse_input.rotationUsage
    if deesse_input.rotationUsage == 1:
        if deesse_input.rotationAzimuthLocal:
            mpds_siminput.rotationAzimuthImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationAzimuth)
            mpds_siminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_siminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_siminput.rotationAzimuthValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationAzimuthValue, 0,
                np.asarray(deesse_input.rotationAzimuth).reshape(1))

        if deesse_input.rotationDipLocal:
            mpds_siminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationDip)
            mpds_siminput.rotationDipImage = img_py2C(im)

        else:
            mpds_siminput.rotationDipImageFlag = deesse.FALSE
            mpds_siminput.rotationDipValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationDipValue, 0,
                np.asarray(deesse_input.rotationDip).reshape(1))

        if deesse_input.rotationPlungeLocal:
            mpds_siminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationPlunge)
            mpds_siminput.rotationPlungeImage = img_py2C(im)

        else:
            mpds_siminput.rotationPlungeImageFlag = deesse.FALSE
            mpds_siminput.rotationPlungeValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationPlungeValue, 0,
                np.asarray(deesse_input.rotationPlunge).reshape(1))

    elif deesse_input.rotationUsage == 2:
        if deesse_input.rotationAzimuthLocal:
            mpds_siminput.rotationAzimuthImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationAzimuth)
            mpds_siminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_siminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_siminput.rotationAzimuthValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationAzimuthValue, 0,
                np.asarray(deesse_input.rotationAzimuth).reshape(2))

        if deesse_input.rotationDipLocal:
            mpds_siminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationDip)
            mpds_siminput.rotationDipImage = img_py2C(im)

        else:
            mpds_siminput.rotationDipImageFlag = deesse.FALSE
            mpds_siminput.rotationDipValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationDipValue, 0,
                np.asarray(deesse_input.rotationDip).reshape(2))

        if deesse_input.rotationPlungeLocal:
            mpds_siminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationPlunge)
            mpds_siminput.rotationPlungeImage = img_py2C(im)

        else:
            mpds_siminput.rotationPlungeImageFlag = deesse.FALSE
            mpds_siminput.rotationPlungeValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationPlungeValue, 0,
                np.asarray(deesse_input.rotationPlunge).reshape(2))

    # mpds_siminput.trainValueRangeExtensionMax
    mpds_siminput.trainValueRangeExtensionMax = deesse_input.expMax

    # mpds_siminput.normalizingType
    if deesse_input.normalizingType == 'linear':
        mpds_siminput.normalizingType = deesse.NORMALIZING_LINEAR
    elif deesse_input.normalizingType == 'uniform':
        mpds_siminput.normalizingType = deesse.NORMALIZING_UNIFORM
    elif deesse_input.normalizingType == 'normal':
        mpds_siminput.normalizingType = deesse.NORMALIZING_NORMAL
    else:
        print ('ERROR: normalizing type unknown')
        return

    # mpds_siminput.searchNeighborhoodParameters
    mpds_siminput.searchNeighborhoodParameters = deesse.new_MPDS_SEARCHNEIGHBORHOODPARAMETERS_array(int(deesse_input.nv))
    for i, sn in enumerate(deesse_input.searchNeighborhoodParameters):
        sn_c = deesse.malloc_MPDS_SEARCHNEIGHBORHOODPARAMETERS()
        deesse.MPDSInitSearchNeighborhoodParameters(sn_c)
        if sn.radiusMode == 'large_default':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT
        elif sn.radiusMode == 'ti_range_default':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT
        elif sn.radiusMode == 'ti_range':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE
        elif sn.radiusMode == 'ti_range_xy':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY
        elif sn.radiusMode == 'ti_range_xz':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ
        elif sn.radiusMode == 'ti_range_yz':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ
        elif sn.radiusMode == 'ti_range_xyz':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ
        elif sn.radiusMode == 'manual':
            sn_c.radiusMode = deesse.SEARCHNEIGHBORHOOD_RADIUS_MANUAL
        sn_c.rx = sn.rx
        sn_c.ry = sn.ry
        sn_c.rz = sn.rz
        if sn.anisotropyRatioMode == 'one':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE
        elif sn.anisotropyRatioMode == 'radius':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS
        elif sn.anisotropyRatioMode == 'radius_xy':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY
        elif sn.anisotropyRatioMode == 'radius_xz':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ
        elif sn.anisotropyRatioMode == 'radius_yz':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ
        elif sn.anisotropyRatioMode == 'radius_xyz':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ
        elif sn.anisotropyRatioMode == 'manual':
            sn_c.anisotropyRatioMode = deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_MANUAL
        sn_c.ax = sn.ax
        sn_c.ay = sn.ay
        sn_c.az = sn.az
        sn_c.angle1 = sn.angle1
        sn_c.angle2 = sn.angle2
        sn_c.angle3 = sn.angle3
        sn_c.power = sn.power
        deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_setitem(
            mpds_siminput.searchNeighborhoodParameters, i, sn_c)

    # mpds_siminput.nneighboringNode
    mpds_siminput.nneighboringNode = deesse.new_int_array(int(deesse_input.nv))
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.nneighboringNode, 0,
        np.asarray(deesse_input.nneighboringNode, dtype='intc').reshape(int(deesse_input.nv)))

    # mpds_siminput.maxPropInequalityNode
    mpds_siminput.maxPropInequalityNode = deesse.new_double_array(int(deesse_input.nv))
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.maxPropInequalityNode, 0,
        np.asarray(deesse_input.maxPropInequalityNode).reshape(int(deesse_input.nv)))

    # mpds_siminput.neighboringNodeDensity
    mpds_siminput.neighboringNodeDensity = deesse.new_double_array(int(deesse_input.nv))
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.neighboringNodeDensity, 0,
        np.asarray(deesse_input.neighboringNodeDensity).reshape(int(deesse_input.nv)))

    # mpds_siminput.rescalingMode
    mpds_siminput.rescalingMode = deesse.new_MPDS_RESCALINGMODE_array(int(deesse_input.nv))
    for i, m in enumerate(deesse_input.rescalingMode):
        if m == 'none':
            deesse.MPDS_RESCALINGMODE_array_setitem(mpds_siminput.rescalingMode, i, deesse.RESCALING_NONE)
        elif m == 'min_max':
            deesse.MPDS_RESCALINGMODE_array_setitem(mpds_siminput.rescalingMode, i, deesse.RESCALING_MIN_MAX)
        elif m == 'mean_length':
            deesse.MPDS_RESCALINGMODE_array_setitem(mpds_siminput.rescalingMode, i, deesse.RESCALING_MEAN_LENGTH)
        else:
            print ('ERROR: rescaling mode unknown')
            return

    # mpds_simInput.rescalingTargetMin
    mpds_siminput.rescalingTargetMin = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMin, 0,
        np.asarray(deesse_input.rescalingTargetMin).reshape(int(deesse_input.nv)))

    # mpds_simInput.rescalingTargetMax
    mpds_siminput.rescalingTargetMax = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMax, 0,
        np.asarray(deesse_input.rescalingTargetMax).reshape(int(deesse_input.nv)))

    # mpds_simInput.rescalingTargetMean
    mpds_siminput.rescalingTargetMean = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMean, 0,
        np.asarray(deesse_input.rescalingTargetMean).reshape(int(deesse_input.nv)))

    # mpds_simInput.rescalingTargetLength
    mpds_siminput.rescalingTargetLength = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetLength, 0,
        np.asarray(deesse_input.rescalingTargetLength).reshape(int(deesse_input.nv)))

    # mpds_siminput.relativeDistanceFlag
    deesse.mpds_set_relativeDistanceFlag(mpds_siminput, np.array([int(i) for i in deesse_input.relativeDistanceFlag], dtype='intc'))

    # mpds_siminput.distanceType
    mpds_siminput.distanceType = deesse.new_int_array(int(deesse_input.nv))
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.distanceType, 0,
        np.asarray(deesse_input.distanceType, dtype='intc').reshape(int(deesse_input.nv)))

    # mpds_siminput.powerLpDistance
    mpds_siminput.powerLpDistance = deesse.new_double_array(int(deesse_input.nv))
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.powerLpDistance, 0,
        np.asarray(deesse_input.powerLpDistance).reshape(int(deesse_input.nv)))

    # mpds_siminput.powerLpDistanceInv
    mpds_siminput.powerLpDistanceInv = deesse.new_double_array(int(deesse_input.nv))
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.powerLpDistanceInv, 0,
        np.asarray(deesse_input.powerLpDistanceInv).reshape(int(deesse_input.nv)))

    # mpds_siminput.conditioningWeightFactor
    mpds_siminput.conditioningWeightFactor = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.conditioningWeightFactor, 0,
        np.asarray(deesse_input.conditioningWeightFactor).reshape(int(deesse_input.nv)))

    # mpds_siminput.simAndPathParameters
    # ... simType
    mpds_siminput.simAndPathParameters = deesse.malloc_MPDS_SIMANDPATHPARAMETERS()
    deesse.MPDSInitSimAndPathParameters(mpds_siminput.simAndPathParameters)
    if deesse_input.simType == 'sim_one_by_one':
        mpds_siminput.simAndPathParameters.simType = deesse.SIM_ONE_BY_ONE
    elif deesse_input.simType == 'sim_variable_vector':
        mpds_siminput.simAndPathParameters.simType = deesse.SIM_VARIABLE_VECTOR
    else:
        print ('ERROR: simulation type unknown')
        return

    # ... simPathType
    if deesse_input.simPathType == 'random':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_RANDOM
    elif deesse_input.simPathType == 'random_hd_distance_pdf':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_RANDOM_HD_DISTANCE_PDF
        mpds_siminput.simAndPathParameters.strength = deesse_input.simPathStrength
    elif deesse_input.simPathType == 'random_hd_distance_sort':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SORT
        mpds_siminput.simAndPathParameters.strength = deesse_input.simPathStrength
    elif deesse_input.simPathType == 'random_hd_distance_sum_pdf':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SUM_PDF
        mpds_siminput.simAndPathParameters.pow = deesse_input.simPathPower
        mpds_siminput.simAndPathParameters.strength = deesse_input.simPathStrength
    elif deesse_input.simPathType == 'random_hd_distance_sum_sort':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SUM_SORT
        mpds_siminput.simAndPathParameters.pow = deesse_input.simPathPower
        mpds_siminput.simAndPathParameters.strength = deesse_input.simPathStrength
    elif deesse_input.simPathType == 'unilateral':
        mpds_siminput.simAndPathParameters.pathType = deesse.PATH_UNILATERAL
        mpds_siminput.simAndPathParameters.unilateralOrderLength = len(deesse_input.simPathUnilateralOrder)
        mpds_siminput.simAndPathParameters.unilateralOrder = deesse.new_int_array(len(deesse_input.simPathUnilateralOrder))
        deesse.mpds_set_int_vector_from_array(
            mpds_siminput.simAndPathParameters.unilateralOrder, 0,
            np.asarray(deesse_input.simPathUnilateralOrder, dtype='intc').reshape(len(deesse_input.simPathUnilateralOrder)))
    else:
        print ('ERROR: path type unknown')
        return

    # mpds_siminput.distanceThreshold
    mpds_siminput.distanceThreshold = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.distanceThreshold, 0,
        np.asarray(deesse_input.distanceThreshold).reshape(int(deesse_input.nv)))

    # mpds_siminput.softProbability ...
    mpds_siminput.softProbability = deesse.new_MPDS_SOFTPROBABILITY_array(int(deesse_input.nv))

    # ... for each variable ...
    for i, sp in enumerate(deesse_input.softProbability):
        sp_c = deesse.malloc_MPDS_SOFTPROBABILITY()
        deesse.MPDSInitSoftProbability(sp_c)

        # ... probabilityConstraintUsage
        sp_c.probabilityConstraintUsage = sp.probabilityConstraintUsage
        if sp.probabilityConstraintUsage == 0:
            deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_siminput.softProbability, i, sp_c)
            continue

        # ... classOfValues
        sp_c.classOfValues = classInterval2classOfValues(sp.classInterval)
        # sp_c.classOfValues = deesse.malloc_MPDS_CLASSOFVALUES()
        # deesse.MPDSInitClassOfValues(sp_c.classOfValues)
        # sp_c.classOfValues.nclass = sp.nclass
        # sp_c.classOfValues.ninterval = deesse.new_int_array(sp.nclass)
        # sp_c.classOfValues.intervalInf = deesse.new_realp_array(sp.nclass)
        # sp_c.classOfValues.intervalSup = deesse.new_realp_array(sp.nclass)
        # for j, ci in enumerate(sp.classInterval):
        #     nint = len(ci) # = ci.shape[0]
        #     deesse.int_array_setitem(sp_c.classOfValues.ninterval, j, nint)
        #     intInf_c = deesse.new_real_array(nint)
        #     deesse.mpds_set_real_vector_from_array(intInf_c, 0, ci[:,0])
        #     deesse.realp_array_setitem(sp_c.classOfValues.intervalInf, j, intInf_c)
        #     intSup_c = deesse.new_real_array(nint)
        #     deesse.mpds_set_real_vector_from_array(intSup_c, 0, ci[:,1])
        #     deesse.realp_array_setitem(sp_c.classOfValues.intervalSup, j, intSup_c)


        if sp.probabilityConstraintUsage == 1:
            # ... globalPdf
            sp_c.globalPdf = deesse.new_real_array(sp.nclass)
            deesse.mpds_set_real_vector_from_array(sp_c.globalPdf, 0,
                np.asarray(sp.globalPdf).reshape(sp.nclass))

        elif sp.probabilityConstraintUsage == 2:
            # ... localPdf
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=sp.nclass, val=sp.localPdf)
            sp_c.localPdfImage = img_py2C(im)

            # ... localPdfSupportRadius
            sp_c.localPdfSupportRadius = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(sp_c.localPdfSupportRadius, 0,
                np.asarray(sp.localPdfSupportRadius).reshape(1))

            # ... localCurrentPdfComputation
            sp_c.localCurrentPdfComputation = sp.localCurrentPdfComputation

        # ... comparingPdfMethod
        sp_c.comparingPdfMethod = sp.comparingPdfMethod

        # ... deactivationDistance
        sp_c.deactivationDistance = sp.deactivationDistance

        # ... probabilityConstraintThresholdType
        sp_c.probabilityConstraintThresholdType = sp.probabilityConstraintThresholdType

        # ... constantThreshold
        sp_c.constantThreshold = sp.constantThreshold

        if sp.probabilityConstraintThresholdType == 1:
            # ... dynamicThresholdParameters
            sp_c.dynamicThresholdParameters = deesse.new_real_array(7)
            deesse.mpds_set_real_vector_from_array(sp_c.dynamicThresholdParameters, 0,
                np.asarray(sp.dynamicThresholdParameters).reshape(7))

        deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_siminput.softProbability, i, sp_c)

    # mpds_siminput.connectivity ...
    mpds_siminput.connectivity = deesse.new_MPDS_CONNECTIVITY_array(int(deesse_input.nv))

    for i, co in enumerate(deesse_input.connectivity):
        co_c = deesse.malloc_MPDS_CONNECTIVITY()
        deesse.MPDSInitConnectivity(co_c)

        # ... connectivityConstraintUsage
        co_c.connectivityConstraintUsage = co.connectivityConstraintUsage
        if co.connectivityConstraintUsage == 0:
            deesse.MPDS_CONNECTIVITY_array_setitem(mpds_siminput.connectivity, i, co_c)
            continue

        # ... connectivityType
        if co.connectivityType == 'connect_face':
            co_c.connectivityType = deesse.CONNECT_FACE
        elif co.connectivityType == 'connect_face_edge':
            co_c.connectivityType = deesse.CONNECT_FACE_EDGE
        elif co.connectivityType == 'connect_face_edge_corner':
            co_c.connectivityType = deesse.CONNECT_FACE_EDGE_CORNER
        else:
            print ('ERROR: connectivity type unknown')
            return

        # ... varName
        deesse.mpds_set_connectivity_varname(co_c, co.varname)

        # ... classOfValues
        co_c.classOfValues = classInterval2classOfValues(co.classInterval)

        # ... tiAsRefFlag
        if co.tiAsRefFlag:
            co_c.tiAsRefFlag = deesse.TRUE
        else:
            co_c.tiAsRefFlag = deesse.FALSE

        if not co.tiAsRefFlag:
            # ... refConnectivityImage
            im = img.copyImg(co.refConnectivityImage)
            im.extract_var([co.refConnectivityVarIndex])
            co_c.refConnectivityImage = img_py2C(im)

        # ... deactivationDistance
        co_c.deactivationDistance = co.deactivationDistance

        # ... threshold
        co_c.threshold = co.threshold

        deesse.MPDS_CONNECTIVITY_array_setitem(mpds_siminput.connectivity, i, co_c)

    # mpds_siminput.blockData ...
    mpds_siminput.blockData = deesse.new_MPDS_BLOCKDATA_array(int(deesse_input.nv))
    # ... for each variable ...
    for i, bd in enumerate(deesse_input.blockData):
        bd_c = deesse.malloc_MPDS_BLOCKDATA()
        deesse.MPDSInitBlockData(bd_c)

        # ... blockDataUsage
        bd_c.blockDataUsage = bd.blockDataUsage
        if bd.blockDataUsage == 0:
            deesse.MPDS_BLOCKDATA_array_setitem(mpds_siminput.blockData, i, bd_c)
            continue

        # ... nblock
        bd_c.nblock = bd.nblock

        # ... nnode, ix, iy, iz
        bd_c.nnode = deesse.new_int_array(bd.nblock)
        bd_c.ix = deesse.new_intp_array(bd.nblock)
        bd_c.iy = deesse.new_intp_array(bd.nblock)
        bd_c.iz = deesse.new_intp_array(bd.nblock)

        for j, ni in enumerate(bd.nodeIndex):
            nn = len(ni) # = ni.shape[0]
            deesse.int_array_setitem(bd_c.nnode, j, nn)
            ix_c = deesse.new_int_array(nn)
            deesse.mpds_set_int_vector_from_array(ix_c, 0, np.asarray(ni[:,0], dtype='intc'))
            deesse.intp_array_setitem(bd_c.ix, j, ix_c)
            iy_c = deesse.new_int_array(nn)
            deesse.mpds_set_int_vector_from_array(iy_c, 0, np.asarray(ni[:,1], dtype='intc'))
            deesse.intp_array_setitem(bd_c.iy, j, iy_c)
            iz_c = deesse.new_int_array(nn)
            deesse.mpds_set_int_vector_from_array(iz_c, 0, np.asarray(ni[:,2], dtype='intc'))
            deesse.intp_array_setitem(bd_c.iz, j, iz_c)

        # ... value
        bd_c.value = deesse.new_real_array(bd.nblock)
        deesse.mpds_set_real_vector_from_array(bd_c.value, 0,
            np.asarray(bd.value).reshape(bd.nblock))

        # ... tolerance
        bd_c.tolerance = deesse.new_real_array(bd.nblock)
        deesse.mpds_set_real_vector_from_array(bd_c.tolerance, 0,
            np.asarray(bd.tolerance).reshape(bd.nblock))

        # ... activatePropMin
        bd_c.activatePropMin = deesse.new_real_array(bd.nblock)
        deesse.mpds_set_real_vector_from_array(bd_c.activatePropMin, 0,
            np.asarray(bd.activatePropMin).reshape(bd.nblock))

        # ... activatePropMax
        bd_c.activatePropMax = deesse.new_real_array(bd.nblock)
        deesse.mpds_set_real_vector_from_array(bd_c.activatePropMax, 0,
            np.asarray(bd.activatePropMax).reshape(bd.nblock))

        deesse.MPDS_BLOCKDATA_array_setitem(mpds_siminput.blockData, i, bd_c)

    # mpds_siminput.maxScanFraction
    mpds_siminput.maxScanFraction = deesse.new_double_array(deesse_input.nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.maxScanFraction, 0,
            np.asarray(deesse_input.maxScanFraction).reshape(deesse_input.nTI))

    # mpds_siminput.pyramidGeneralParameters ...
    mpds_siminput.pyramidGeneralParameters = deesse.malloc_MPDS_PYRAMIDGENERALPARAMETERS()
    deesse.MPDSInitPyramidGeneralParameters(mpds_siminput.pyramidGeneralParameters)

    # ... npyramidLevel
    nl = int(deesse_input.pyramidGeneralParameters.npyramidLevel)
    mpds_siminput.pyramidGeneralParameters.npyramidLevel = nl

    # ... pyramidSimulationMode
    if deesse_input.pyramidGeneralParameters.pyramidSimulationMode == 'hierarchical':
        mpds_siminput.pyramidGeneralParameters.pyramidSimulationMode = deesse.PYRAMID_SIM_HIERARCHICAL
    elif deesse_input.pyramidGeneralParameters.pyramidSimulationMode == 'hierarchical_using_expansion':
        mpds_siminput.pyramidGeneralParameters.pyramidSimulationMode = deesse.PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION
    elif deesse_input.pyramidGeneralParameters.pyramidSimulationMode == 'all_level_one_by_one':
        mpds_siminput.pyramidGeneralParameters.pyramidSimulationMode = deesse.PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE
    else:
        mpds_siminput.pyramidGeneralParameters.pyramidSimulationMode = deesse.PYRAMID_SIM_NONE

    if nl > 0:
        # ... kx
        mpds_siminput.pyramidGeneralParameters.kx = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.kx, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.kx, dtype='intc').reshape(nl))

        # ... ky
        mpds_siminput.pyramidGeneralParameters.ky = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.ky, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.ky, dtype='intc').reshape(nl))

        # ... kz
        mpds_siminput.pyramidGeneralParameters.kz = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.kz, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.kz, dtype='intc').reshape(nl))

        # ... factorNneighboringNode and factorDistanceThreshold ...
        if deesse_input.pyramidGeneralParameters.pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            nn = 4*nl + 1
        else: # pyramidSimulationMode == 'all_level_one_by_one'
            nn = nl + 1

        # ... factorNneighboringNode
        mpds_siminput.pyramidGeneralParameters.factorNneighboringNode = deesse.new_double_array(nn)
        deesse.mpds_set_double_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.factorNneighboringNode, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.factorNneighboringNode).reshape(nn))

        # ... factorDistanceThreshold
        mpds_siminput.pyramidGeneralParameters.factorDistanceThreshold = deesse.new_real_array(nn)
        deesse.mpds_set_real_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.factorDistanceThreshold, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.factorDistanceThreshold).reshape(nn))

        # ... factorMaxScanFraction
        mpds_siminput.pyramidGeneralParameters.factorMaxScanFraction = deesse.new_double_array(nl+1)
        deesse.mpds_set_double_vector_from_array(
            mpds_siminput.pyramidGeneralParameters.factorMaxScanFraction, 0,
                np.asarray(deesse_input.pyramidGeneralParameters.factorMaxScanFraction).reshape(nl+1))

    # mpds_siminput.pyramidParameters ...
    mpds_siminput.pyramidParameters = deesse.new_MPDS_PYRAMIDPARAMETERS_array(int(deesse_input.nv))

    # ... for each variable ...
    for i, pp in enumerate(deesse_input.pyramidParameters):
        pp_c = deesse.malloc_MPDS_PYRAMIDPARAMETERS()
        deesse.MPDSInitPyramidParameters(pp_c)

        # ... nlevel
        pp_c.nlevel = int(pp.nlevel)

        # ... pyramidType
        if pp.pyramidType == 'none':
            pp_c.pyramidType = deesse.PYRAMID_NONE
        elif pp.pyramidType == 'continuous':
            pp_c.pyramidType = deesse.PYRAMID_CONTINUOUS
        elif pp.pyramidType == 'categorical_auto':
            pp_c.pyramidType = deesse.PYRAMID_CATEGORICAL_AUTO
        elif pp.pyramidType == 'categorical_custom':
            pp_c.pyramidType = deesse.PYRAMID_CATEGORICAL_CUSTOM
        elif pp.pyramidType == 'categorical_to_continuous':
            pp_c.pyramidType = deesse.PYRAMID_CATEGORICAL_TO_CONTINUOUS
        else:
            pp_c.pyramidType = deesse.PYRAMID_NONE

        if pp.pyramidType == 'categorical_custom':
            # ... classOfValues
            pp_c.classOfValues = classInterval2classOfValues(pp.classInterval)

        deesse.MPDS_PYRAMIDPARAMETERS_array_setitem(mpds_siminput.pyramidParameters, i, pp_c)

    # mpds_siminput.tolerance
    mpds_siminput.tolerance = deesse_input.tolerance

    # mpds_siminput.npostProcessingPathMax
    mpds_siminput.npostProcessingPathMax = deesse_input.npostProcessingPathMax

    # mpds_siminput.postProcessingNneighboringNode
    mpds_siminput.postProcessingNneighboringNode = deesse.new_int_array(int(deesse_input.nv))
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.postProcessingNneighboringNode, 0,
            np.asarray(deesse_input.postProcessingNneighboringNode, dtype='intc').reshape(int(deesse_input.nv)))

    # mpds_siminput.postProcessingNeighboringNodeDensity
    mpds_siminput.postProcessingNeighboringNodeDensity = deesse.new_double_array(int(deesse_input.nv))
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.postProcessingNeighboringNodeDensity, 0,
            np.asarray(deesse_input.postProcessingNeighboringNodeDensity).reshape(int(deesse_input.nv)))

    # mpds_siminput.postProcessingDistanceThreshold
    mpds_siminput.postProcessingDistanceThreshold = deesse.new_real_array(int(deesse_input.nv))
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.postProcessingDistanceThreshold, 0,
            np.asarray(deesse_input.postProcessingDistanceThreshold).reshape(int(deesse_input.nv)))

    # mpds_siminput.postProcessingMaxScanFraction
    mpds_siminput.postProcessingMaxScanFraction = deesse.new_double_array(deesse_input.nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.postProcessingMaxScanFraction, 0,
            np.asarray(deesse_input.postProcessingMaxScanFraction).reshape(deesse_input.nTI))

    # mpds_siminput.postProcessingTolerance
    mpds_siminput.postProcessingTolerance = deesse_input.postProcessingTolerance

    # mpds_siminput.seed
    mpds_siminput.seed = int(deesse_input.seed)

    # mpds_siminput.seedIncrement
    mpds_siminput.seedIncrement = int(deesse_input.seedIncrement)

    # mpds_siminput.nrealization
    mpds_siminput.nrealization = int(deesse_input.nrealization)

    return mpds_siminput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_output_C2py(mpds_simoutput, mpds_progressMonitor):
    """
    Get deesse output for python from C.

    :param mpds_simoutput:  (MPDS_SIMOUTPUT *) simulation output - (C struct)
    :param mpds_progressMonitor:
                            (MPDS_PROGRESSMONITOR *) progress monitor - (C struct)

    :return deesse_output:  (dict)
            {'sim':sim,
             'path':path,
             'error':error,
             'tiGridNode':tiGridNode,
             'tiIndex':tiIndex,
             'nwarning':nwarning,
             'warnings':warnings}
        With nreal = mpds_simOutput->nreal:
        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation
                            (mpds_simoutput->outputSimImage[i])
                    (sim is None if mpds_simoutput->outputSimImage is NULL)
        path:   (1-dimensional array of Img (class) of size nreal or None)
                    path[i]: path index map for the i-th realisation
                             (mpds_simoutput->outputPathIndexImage[i])
                    (path is None if mpds_simoutput->outputPathIndexImage is NULL)
        error:   (1-dimensional array of Img (class) of size nreal or None)
                    error[i]: error map for the i-th realisation
                              (mpds_simoutput->outputErrorImage[i])
                    (error is None if mpds_simoutput->outputErrorImage is NULL)
        tiGridNode:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiGridNode[i]: TI grid node index map for the i-th realisation
                             (mpds_simoutput->outputTiGridNodeIndexImage[i])
                    (tiGridNode is None if mpds_simoutput->outputTiGridNodeIndexImage is NULL)
        tiIndex:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiIndex[i]: TI index map for the i-th realisation
                             (mpds_simoutput->outputTiIndexImage[i])
                    (tiIndex is None if mpds_simoutput->outputTiIndexImage is NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Initialization
    sim, path, error, tiGridNode, tiIndex = None, None, None, None, None
    nwarning, warnings = None, None

    if mpds_simoutput.nreal:
        nreal = mpds_simoutput.nreal

        if mpds_simoutput.nvarSimPerReal:
            # Retrieve the list of simulated image
            im = img_C2py(mpds_simoutput.outputSimImage)

            nv = mpds_simoutput.nvarSimPerReal
            k = 0
            sim = []
            for i in range(nreal):
                sim.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                               sx=im.sx, sy=im.sy, sz=im.sz,
                               ox=im.ox, oy=im.oy, oz=im.oz,
                               nv=nv, val=im.val[k:(k+nv),...],
                               varname=im.varname[k:(k+nv)]))
                k = k + nv

            sim = np.asarray(sim).reshape(nreal)

        if mpds_simoutput.nvarPathIndexPerReal:
            # Retrieve the list of path index image
            im = img_C2py(mpds_simoutput.outputPathIndexImage)

            nv = mpds_simoutput.nvarPathIndexPerReal
            k = 0
            path = []
            for i in range(nreal):
                path.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)]))
                k = k + nv

            path = np.asarray(path).reshape(nreal)

        if mpds_simoutput.nvarErrorPerReal:
            # Retrieve the list of error image
            im = img_C2py(mpds_simoutput.outputErrorImage)

            nv = mpds_simoutput.nvarErrorPerReal
            k = 0
            error = []
            for i in range(nreal):
                error.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                 sx=im.sx, sy=im.sy, sz=im.sz,
                                 ox=im.ox, oy=im.oy, oz=im.oz,
                                 nv=nv, val=im.val[k:(k+nv),...],
                                 varname=im.varname[k:(k+nv)]))
                k = k + nv

            error = np.asarray(error).reshape(nreal)

        if mpds_simoutput.nvarTiGridNodeIndexPerReal:
            # Retrieve the list of TI grid node index image
            im = img_C2py(mpds_simoutput.outputTiGridNodeIndexImage)

            nv = mpds_simoutput.nvarTiGridNodeIndexPerReal
            k = 0
            tiGridNode = []
            for i in range(nreal):
                tiGridNode.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)]))
                k = k + nv

            tiGridNode = np.asarray(tiGridNode).reshape(nreal)

        if mpds_simoutput.nvarTiIndexPerReal:
            # Retrieve the list of TI index image
            im = img_C2py(mpds_simoutput.outputTiIndexImage)

            nv = mpds_simoutput.nvarTiIndexPerReal
            k = 0
            tiIndex = []
            for i in range(nreal):
                tiIndex.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)]))
                k = k + nv

            tiIndex = np.asarray(tiIndex).reshape(nreal)

    nwarning = mpds_progressMonitor.nwarning
    warnings = []
    if mpds_progressMonitor.nwarningNumber:
        tmp = np.zeros(mpds_progressMonitor.nwarningNumber, dtype='int32') # 'int32' for C-compatibility
        deesse.mpds_get_array_from_int_vector(mpds_progressMonitor.warningNumberList, 0, tmp)
        warningNumberList = np.asarray(tmp, dtype='int') # 'int' or equivalently 'int64'
        for iwarn in warningNumberList:
            warning_message = deesse.mpds_get_warning_message(int(iwarn)) # int() required!
            warning_message = warning_message.replace('\n', '')
            warnings.append(warning_message)

    return {'sim':sim, 'path':path, 'error':error, 'tiGridNode':tiGridNode, 'tiIndex':tiIndex, 'nwarning':nwarning, 'warnings':warnings}
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseRun(deesse_input, nthreads=-1, verbose=2):
    """
    Launches deesse.

    :param deesse_input:
                (DeesseInput (class)): deesse input parameter (python)
    :param nthreads:
                (int) number of thread(s) to use for deesse (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the deesse run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return deesse_output:
        (dict)
                {'sim':sim,
                 'path':path,
                 'error':error,
                 'tiGridNode':tiGridNode,
                 'tiIndex':tiIndex,
                 'nwarning':nwarning,
                 'warnings':warnings}
            With nreal = deesse_input.nrealization:
            sim:    (1-dimensional array of Img (class) of size nreal or None)
                        sim[i]: i-th realisation
                        (sim is None if no simulation is retrieved)
            path:   (1-dimensional array of Img (class) of size nreal or None)
                        path[i]: path index map for the i-th realisation
                        (path is None if no path index map is retrieved)
            error:   (1-dimensional array of Img (class) of size nreal or None)
                        error[i]: error map for the i-th realisation
                        (error is None if no error map is retrieved)
            tiGridNode:
                    (1-dimensional array of Img (class) of size nreal or None)
                        tiGridNode[i]: TI grid node index map for the i-th realisation
                        (tiGridNode is None if no TI grid node index map is retrieved)
            tiIndex:
                    (1-dimensional array of Img (class) of size nreal or None)
                        tiIndex[i]: TI index map for the i-th realisation
                        (tiIndex is None if no TI index map is retrieved)
            nwarning:
                    (int) total number of warning(s) encountered
                        (same warnings can be counted several times)
            warnings:
                    (list of strings) list of distinct warnings encountered
                        (can be empty)
    """

    # Convert deesse input from python to C
    if not deesse_input.ok:
        print('ERROR: check deesse input')
        return

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('DeeSse running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
        sys.stdout.flush() # so that the previous print is flushed before launching deesse...

    # Convert deesse input from python to C
    try:
        mpds_siminput = deesse_input_py2C(deesse_input)
    except:
        print('ERROR: unable to convert deesse input from python to C...')
        return

    if mpds_siminput is None:
        print('ERROR: unable to convert deesse input from python to C...')
        return

    # Allocate mpds_simoutput
    mpds_simoutput = deesse.malloc_MPDS_SIMOUTPUT()

    # Init mpds_simoutput
    deesse.MPDSInitSimOutput(mpds_simoutput)

    # Set progress monitor
    mpds_progressMonitor = deesse.malloc_MPDS_PROGRESSMONITOR()
    deesse.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to deesse.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # Launch deesse
    # err = deesse.MPDSSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = deesse.MPDSOMPSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: deesse input
    deesse.MPDSFreeSimInput(mpds_siminput)
    #deesse.MPDSFree(mpds_siminput)
    deesse.free_MPDS_SIMINPUT(mpds_siminput)

    if err:
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        deesse_output = None
    else:
        deesse_output = deesse_output_C2py(mpds_simoutput, mpds_progressMonitor)

    # Free memory on C side: simulation output
    deesse.MPDSFreeSimOutput(mpds_simoutput)
    #deesse.MPDSFree (mpds_simoutput)
    deesse.free_MPDS_SIMOUTPUT(mpds_simoutput)

    # Free memory on C side: progress monitor
    #deesse.MPDSFree(mpds_progressMonitor)
    deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and deesse_output:
        print('DeeSse run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and deesse_output and deesse_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(deesse_output['nwarning']))
        for i, warning_message in enumerate(deesse_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return deesse_output
# ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------
# # Note: the three following functions:
# #          deesseRunC, deesseRun_sp, and deesseRun_mp
# #       are based on multiprocessing package, but as their use implies
# #       unwanted behaviours in notebook, they are commented...
# # ----------------------------------------------------------------------------
# def deesseRunC(deesse_input, nthreads, verbose):
#     """
#     Launches deesse in C (single process).
#
#     :param deesse_input:
#                 (DeesseInput (class)): deesse input parameter (python)
#     :param nthreads:
#                 (int) number of thread(s) to use for deesse (C), nthreads > 0
#     :param verbose:
#                 (int) indicates what is displayed during the deesse run:
#                     - 0: nothing
#                     - 1: warning only
#                     - 2 (or >1): warning and progress
#
#     :return (deesse_output, err, err_message):
#         deesse_output: (dict)
#                 {'sim':sim,
#                  'path':path,
#                  'error':error,
#                  'tiGridNode':tiGridNode,
#                  'tiIndex':tiIndex,
#                  'nwarning':nwarning,
#                  'warnings':warnings}
#             With nreal = deesse_input.nrealization:
#             sim:    (1-dimensional array of Img (class) of size nreal or None)
#                         sim[i]: i-th realisation
#                         (sim is None if no simulation is retrieved)
#             path:   (1-dimensional array of Img (class) of size nreal or None)
#                         path[i]: path index map for the i-th realisation
#                         (path is None if no path index map is retrieved)
#             error:   (1-dimensional array of Img (class) of size nreal or None)
#                         error[i]: error map for the i-th realisation
#                         (error is None if no error map is retrieved)
#             tiGridNode:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiGridNode[i]: TI grid node index map for the i-th realisation
#                         (tiGridNode is None if no TI grid node index map is retrieved)
#             tiIndex:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiIndex[i]: TI index map for the i-th realisation
#                         (tiIndex is None if no TI index map is retrieved)
#             nwarning:
#                     (int) total number of warning(s) encountered
#                         (same warnings can be counted several times)
#             warnings:
#                     (list of strings) list of distinct warnings encountered
#                         (can be empty)
#
#         err: (int) error code:
#             = 0: no error
#             < 0: error resulting from the C function
#             > 0: error intercepted in this function
#
#         err_message: (string) error message
#
#     :note: a call of this function should be isolated in a process (multiprocessing package)
#     """
#
#     # Initialization
#     err = 0
#     err_message = ''
#     deesse_output = None
#
#     # Check number of threads
#     if nthreads <= 0:
#         err = 2
#         err_message = 'ERROR: invalid number of threads...'
#         return (deesse_output, err, err_message)
#
#     # Convert deesse input from python to C
#     try:
#         mpds_siminput = deesse_input_py2C(deesse_input)
#     except:
#         err = 3
#         err_message = 'ERROR: unable to convert deesse input from python to C...'
#         return (deesse_output, err, err_message)
#
#     if mpds_siminput is None:
#         err = 3
#         err_message = 'ERROR: unable to convert deesse input from python to C...'
#         return (deesse_output, err, err_message)
#
#     # Allocate mpds_simoutput
#     mpds_simoutput = deesse.malloc_MPDS_SIMOUTPUT()
#
#     # Init mpds_simoutput
#     deesse.MPDSInitSimOutput(mpds_simoutput)
#
#     # Set progress monitor
#     mpds_progressMonitor = deesse.malloc_MPDS_PROGRESSMONITOR()
#     deesse.MPDSInitProgressMonitor(mpds_progressMonitor)
#
#     # Set function to update progress monitor:
#     # according to deesse.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
#     # the function
#     #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
#     # should be used, but the following function can also be used:
#     #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr: no output
#     #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
#     if verbose == 0:
#         mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr
#     elif verbose == 1:
#         mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
#     else:
#         mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
#
#     # Launch deesse
#     # err = deesse.MPDSSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
#     err = deesse.MPDSOMPSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor, nthreads )
#
#     # Free memory on C side: deesse input
#     deesse.MPDSFreeSimInput(mpds_siminput)
#     #deesse.MPDSFree(mpds_siminput)
#     deesse.free_MPDS_SIMINPUT(mpds_siminput)
#
#     if err:
#         err_message = deesse.mpds_get_error_message(-err)
#         err_message = err_message.replace('\n', '')
#     else:
#         deesse_output = deesse_output_C2py(mpds_simoutput, mpds_progressMonitor)
#
#     # Free memory on C side: simulation output
#     deesse.MPDSFreeSimOutput(mpds_simoutput)
#     #deesse.MPDSFree (mpds_simoutput)
#     deesse.free_MPDS_SIMOUTPUT(mpds_simoutput)
#
#     # Free memory on C side: progress monitor
#     #deesse.MPDSFree(mpds_progressMonitor)
#     deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
#
#     return (deesse_output, err, err_message)
# # ----------------------------------------------------------------------------
#
# # ----------------------------------------------------------------------------
# def deesseRun_sp(deesse_input, nthreads=-1, verbose=2):
#     """
#     Launches deesse through a single process.
#
#     :param deesse_input:
#                 (DeesseInput (class)): deesse input parameter (python)
#     :param nthreads:
#                 (int) number of thread(s) to use for deesse (C),
#                     (nthreads = -n <= 0: for maximal number of threads except n,
#                     but at least 1)
#     :param verbose:
#                 (int) indicates what is displayed during the deesse run:
#                     - 0: nothing
#                     - 1: warning only
#                     - 2 (or >1): warning and progress
#
#     :return deesse_output:
#         (dict)
#                 {'sim':sim,
#                  'path':path,
#                  'error':error,
#                  'tiGridNode':tiGridNode,
#                  'tiIndex':tiIndex,
#                  'nwarning':nwarning,
#                  'warnings':warnings}
#             With nreal = deesse_input.nrealization:
#             sim:    (1-dimensional array of Img (class) of size nreal or None)
#                         sim[i]: i-th realisation
#                         (sim is None if no simulation is retrieved)
#             path:   (1-dimensional array of Img (class) of size nreal or None)
#                         path[i]: path index map for the i-th realisation
#                         (path is None if no path index map is retrieved)
#             error:   (1-dimensional array of Img (class) of size nreal or None)
#                         error[i]: error map for the i-th realisation
#                         (error is None if no error map is retrieved)
#             tiGridNode:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiGridNode[i]: TI grid node index map for the i-th realisation
#                         (tiGridNode is None if no TI grid node index map is retrieved)
#             tiIndex:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiIndex[i]: TI index map for the i-th realisation
#                         (tiIndex is None if no TI index map is retrieved)
#             nwarning:
#                     (int) total number of warning(s) encountered
#                         (same warnings can be counted several times)
#             warnings:
#                     (list of strings) list of distinct warnings encountered
#                         (can be empty)
#     """
#
#     # Convert deesse input from python to C
#     if not deesse_input.ok:
#         print('ERROR: check deesse input')
#         return
#
#     # Set number of threads
#     if nthreads <= 0:
#         nth = max(multiprocessing.cpu_count() + nthreads, 1)
#     else:
#         nth = nthreads
#
#     if verbose >= 2:
#         print('Deesse running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
#         # print('********************************************************************************')
#         # print('DEESSE VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)'.format(deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
#         # print('[{:d} process(es)]'.format(1))
#         # print('********************************************************************************')
#
#     # Run deesse in a pool of 1 worker
#     pool = multiprocessing.Pool(1)
#     out = pool.apply_async(deesseRunC, args=(deesse_input, nth, verbose))
#     # Properly end working process
#     pool.close() # Prevents any more tasks from being submitted to the pool,
#     pool.join()  # then, wait for the worker processes to exit.
#
#     deesse_output, err, err_message = out.get()
#
#     if err:
#         print(err_message)
#         return
#
#     if verbose >= 2:
#         print('Deesse run complete')
#
#     # Show (print) encountered warnings
#     if verbose >= 1 and deesse_output['nwarning']:
#         print('\nWarnings encountered ({} times in all):'.format(deesse_output['nwarning']))
#         for i, warning_message in enumerate(deesse_output['warnings']):
#             print('#{:3d}: {}'.format(i+1, warning_message))
#
#     return deesse_output
# # ----------------------------------------------------------------------------
#
# # ----------------------------------------------------------------------------
# def deesseRun_mp(deesse_input, nprocesses=1, nthreads=None, verbose=2):
#     """
#     Launches deesse through mutliple processes, i.e. nprocesses parallel deesse
#     run(s) using each one nthreads threads will be launched. The set of
#     realizations (specified by deesse_input.nrealization) is distributed in
#     a balanced way over the processes.
#     In terms of resources, this implies the use of nprocesses * nthreads cpu(s).
#
#     :param deesse_input:
#                 (DeesseInput (class)): deesse input parameter (python)
#     :param nprocesses:
#                 (int) number of processes, must be greater than 0
#     :param nthreads:
#                 (int) number of thread(s) to use for each process (C code):
#                     nthreads = None: nthreads is automatically computed as the
#                         maximal integer (but at least 1) such that
#                             nprocesses * nthreads <= nmax
#                         where nmax is the number of threads of the system
#     :param verbose:
#                 (int) indicates what is displayed during the deesse run:
#                     - 0: nothing
#                     - 1: warning only
#                     - 2 (or >1): warning and progress
#
#     :return deesse_output:
#         (dict)
#                 {'sim':sim,
#                  'path':path,
#                  'error':error,
#                  'tiGridNode':tiGridNode,
#                  'tiIndex':tiIndex,
#                  'nwarning':nwarning,
#                  'warnings':warnings}
#             With nreal = deesse_input.nrealization:
#             sim:    (1-dimensional array of Img (class) of size nreal or None)
#                         sim[i]: i-th realisation
#                         (sim is None if no simulation is retrieved)
#             path:   (1-dimensional array of Img (class) of size nreal or None)
#                         path[i]: path index map for the i-th realisation
#                         (path is None if no path index map is retrieved)
#             error:   (1-dimensional array of Img (class) of size nreal or None)
#                         error[i]: error map for the i-th realisation
#                         (error is None if no error map is retrieved)
#             tiGridNode:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiGridNode[i]: TI grid node index map for the i-th realisation
#                         (tiGridNode is None if no TI grid node index map is retrieved)
#             tiIndex:
#                     (1-dimensional array of Img (class) of size nreal or None)
#                         tiIndex[i]: TI index map for the i-th realisation
#                         (tiIndex is None if no TI index map is retrieved)
#             nwarning:
#                     (int) total number of warning(s) encountered
#                         (same warnings can be counted several times)
#             warnings:
#                     (list of strings) list of distinct warnings encountered
#                         (can be empty)
#     """
#
#     # Convert deesse input from python to C
#     if not deesse_input.ok:
#         print('ERROR: check deesse input')
#         return
#
#     # Set number of threads and processes
#     nprocesses = int(nprocesses)
#
#     if nprocesses <= 0:
#         print('ERROR: nprocesses not valid...')
#         return
#
#     if nthreads is None:
#         nth = max(int(np.floor(multiprocessing.cpu_count() / nprocesses)), 1)
#     else:
#         if nthreads <= 0:
#             # nth = max(int(multiprocessing.cpu_count() + nthreads), 1)
#             print('ERROR: nthreads should be positive')
#             return
#         else:
#             nth = int(nthreads)
#
#     if nprocesses * nth > multiprocessing.cpu_count():
#         print('WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')
#
#     # Set the distribution of the realizations over the processes
#     #   real_index_proc[i]: index list of the realization that will be done by i-th process
#     q, r = np.divmod(deesse_input.nrealization, nprocesses)
#     real_index_proc = [i*q + min(i, r) + np.arange(q+(i<r)) for i in range(nprocesses)]
#
#     if verbose >= 2:
#         print('Deesse running in {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]...'.format(nprocesses, deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
#         # print('********************************************************************************')
#         # print('DEESSE VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)'.format(deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
#         # print('[{:d} process(es)]'.format(nprocesses))
#         # print('********************************************************************************')
#
#     # Initialize deesse input for each process
#     deesse_input_proc = [copy.copy(deesse_input) for i in range(nprocesses)]
#     init_seed = deesse_input.seed
#
#     # Set pool of nprocesses workers
#     pool = multiprocessing.Pool(nprocesses)
#     out_pool = []
#
#     for i, input in enumerate(deesse_input_proc):
#         # Adapt deesse input for i-th process
#         input.nrealization = len(real_index_proc[i])
#         if input.nrealization:
#             input.seed = init_seed + int(real_index_proc[i][0]) * input.seedIncrement
#         if i==0:
#             verb = verbose
#         else:
#             verb = min(verbose, 1) # keep only printing of warning if verbose >= 1
#         # Launch deesse (i-th process)
#         out_pool.append(pool.apply_async(deesseRunC, args=(input, nth, verb)))
#
#     # Properly end working process
#     pool.close() # Prevents any more tasks from being submitted to the pool,
#     pool.join()  # then, wait for the worker processes to exit.
#
#     # Get result from each process
#     deesse_result_proc = [p.get() for p in out_pool]
#
#     # Gather results of every process
#     deesse_output_proc = [result[0] for result in deesse_result_proc]
#     deesse_err_proc = [result[1] for result in deesse_result_proc]
#     deesse_err_message_proc = [result[2] for result in deesse_result_proc]
#
#     for output, err, err_message in zip(deesse_output_proc, deesse_err_proc, deesse_err_message_proc):
#         if err:
#             print(err_message)
#             return
#         if output is None:
#             print('ERROR: output cannot be retrieved...')
#             return
#
#     deesse_output = {}
#
#     # 'sim'
#     tmp = [output['sim'] for output in deesse_output_proc if output['sim'] is not None]
#     if tmp:
#         deesse_output['sim'] = np.hstack(tmp)
#         # ... set right index in varname
#         for i in range(deesse_input.nrealization):
#             for j in range(len(deesse_output['sim'][i].varname)):
#                 deesse_output['sim'][i].varname[j] = re.sub('[0-9][0-9][0-9][0-9][0-9]$', '{:05d}'.format(i), deesse_output['sim'][i].varname[j])
#     else:
#         deesse_output['sim'] = None
#
#     # 'path'
#     tmp = [output['path'] for output in deesse_output_proc if output['path'] is not None]
#     if tmp:
#         deesse_output['path'] = np.hstack(tmp)
#         # ... set right index in varname
#         for i in range(deesse_input.nrealization):
#             for j in range(len(deesse_output['path'][i].varname)):
#                 deesse_output['path'][i].varname[j] = re.sub('[0-9][0-9][0-9][0-9][0-9]$', '{:05d}'.format(i), deesse_output['path'][i].varname[j])
#     else:
#         deesse_output['path'] = None
#
#     # 'error'
#     tmp = [output['error'] for output in deesse_output_proc if output['error'] is not None]
#     if tmp:
#         deesse_output['error'] = np.hstack(tmp)
#         # ... set right index in varname
#         for i in range(deesse_input.nrealization):
#             for j in range(len(deesse_output['error'][i].varname)):
#                 deesse_output['error'][i].varname[j] = re.sub('[0-9][0-9][0-9][0-9][0-9]$', '{:05d}'.format(i), deesse_output['error'][i].varname[j])
#     else:
#         deesse_output['error'] = None
#
#     # 'tiGridNode'
#     tmp = [output['tiGridNode'] for output in deesse_output_proc if output['tiGridNode'] is not None]
#     if tmp:
#         deesse_output['tiGridNode'] = np.hstack(tmp)
#         # ... set right index in varname
#         for i in range(deesse_input.nrealization):
#             for j in range(len(deesse_output['tiGridNode'][i].varname)):
#                 deesse_output['tiGridNode'][i].varname[j] = re.sub('[0-9][0-9][0-9][0-9][0-9]$', '{:05d}'.format(i), deesse_output['tiGridNode'][i].varname[j])
#     else:
#         deesse_output['tiGridNode'] = None
#
#     # 'tiIndex'
#     tmp = [output['tiIndex'] for output in deesse_output_proc if output['tiIndex'] is not None]
#     if tmp:
#         deesse_output['tiIndex'] = np.hstack(tmp)
#         # ... set right index in varname
#         for i in range(deesse_input.nrealization):
#             for j in range(len(deesse_output['tiIndex'][i].varname)):
#                 deesse_output['tiIndex'][i].varname[j] = re.sub('[0-9][0-9][0-9][0-9][0-9]$', '{:05d}'.format(i), deesse_output['tiIndex'][i].varname[j])
#     else:
#         deesse_output['tiIndex'] = None
#
#     # 'nwarning'
#     deesse_output['nwarning'] = np.sum([output['nwarning'] for output in deesse_output_proc])
#
#     # 'warnings'
#     tmp = []
#     for output in deesse_output_proc:
#         tmp = tmp + output['warnings'] # concatenation
#     tmp, id = np.unique(tmp, return_index=True)
#     deesse_output['warnings'] = [tmp[id[i]] for i in id]
#
#     if verbose >= 2:
#         print('Deesse run complete')
#
#     # Show (print) encountered warnings
#     if verbose >= 1 and deesse_output['nwarning']:
#         print('\nWarnings encountered ({} times in all):'.format(deesse_output['nwarning']))
#         for i, warning_message in enumerate(deesse_output['warnings']):
#             print('#{:3d}: {}'.format(i+1, warning_message))
#
#     return deesse_output
# # ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def exportDeesseInput(deesse_input,
                      dirname='input_ascii',
                      fileprefix='ds',
                      endofline='\n',
                      suffix_outputSim='',
                      suffix_outputPathIndex='_pathIndex',
                      suffix_outputError='_error',
                      suffix_outputTiGridNodeIndex='_tiGridNodeIndex',
                      suffix_outputTiIndex='_tiIndex',
                      suffix_TI='_ti',
                      suffix_pdfTI='_pdfti',
                      suffix_mask='_mask',
                      suffix_homothetyXRatio='_homothetyXRatio',
                      suffix_homothetyYRatio='_homothetyYRatio',
                      suffix_homothetyZRatio='_homothetyZRatio',
                      suffix_rotationAzimuth='_rotationAzimuth',
                      suffix_rotationDip='_rotationDip',
                      suffix_rotationPlunge='_rotationPlunge',
                      suffix_dataImage='_dataImage',
                      suffix_dataPointSet='_dataPointSet',
                      suffix_localPdf='_localPdf',
                      suffix_refConnectivityImage='_refConnectivityImage',
                      suffix_blockData='_blockData',
                      verbose=2):
    """
    Exports input for deesse as ASCII files (in the directory named <dirname>).
    The command line version of deesse can then be launched from the directory
    <dirname> by using the generated ASCII files.

    :param deesse_input:    (DeesseInput class) deesse input - python
    :param dirname:         (string) name of the directory in which the files will
                                be written; if not existing, it will be created;
                                WARNING: the generated files might erase already
                                existing ones!
    :param fileprefix:      (string) prefix for generated files, the main input
                                file will be <dirname>/<fileprefix>.in
    :param endofline:       (string) end of line string to be used for the deesse
                                input file
    :param suffix_*:        (string) used to name generated files
    :param verbose:         (int) indicates which degree of detail is used when
                                writing comments in the deesse input file
                                - 0: no comment
                                - 1: basic comments
                                - 2: detailed comments
    """
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    inputfile = '{}/{}.in'.format(dirname, fileprefix)

    # open input file
    infid = open(inputfile, "w")

    # Simulation grid
    if verbose > 0:
        infid.write('\
/* SIMULATION GRID (SG) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
{1} {2} {3} // dimension{0}\
{4} {5} {6} // cell size{0}\
{7} {8} {9} // origin{0}\
{0}'.format(endofline,
            deesse_input.nx, deesse_input.ny, deesse_input.nz,
            deesse_input.sx, deesse_input.sy, deesse_input.sz,
            deesse_input.ox, deesse_input.oy, deesse_input.oz))
    else:
        infid.write('\
{1} {2} {3}{0}\
{4} {5} {6}{0}\
{7} {8} {9}{0}\
{0}'.format(endofline,
            deesse_input.nx, deesse_input.ny, deesse_input.nz,
            deesse_input.sx, deesse_input.sy, deesse_input.sz,
            deesse_input.ox, deesse_input.oy, deesse_input.oz))

    # Simulation variables
    if verbose > 0:
        infid.write('\
/* SIMULATION VARIABLES */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Number of simulation variable(s), and for each variable:{0}\
      variable name, output flag (0 / 1), and if output flag is 1: format string (as passed to fprintf).{0}\
   Write DEFAULT_FORMAT for format string to use default format.{0}\
   Example (with 3 variables):{0}\
      3{0}\
      varName1  1  %10.5E{0}\
      varName2  0{0}\
      varName3  1  DEFAULT_FORMAT{0}\
*/{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, deesse_input.nv))

    for vname, vflag in zip(deesse_input.varname, deesse_input.outputVarFlag):
        if vflag:
            infid.write('{1} {2} DEFAULT_FORMAT{0}'.format(endofline, vname, int(vflag)))
        else:
            infid.write('{1} {2}{0}'.format(endofline, vname, int(vflag)))
    infid.write('{0}'.format(endofline))

    # Output settings for simulation
    if verbose > 0:
        infid.write('\
/* OUTPUT SETTINGS FOR SIMULATION */{0}'.format(endofline))

    if verbose == 2:
            infid.write('\
/* Key word and required name(s) or prefix, for output of the realizations:{0}\
      - OUTPUT_SIM_NO_FILE:{0}\
           no file in output{0}\
      - OUTPUT_SIM_ALL_IN_ONE_FILE{0}\
           one file in output (every variable in output (flagged as 1 above),{0}\
           for all realizations will be written),{0}\
           - requires one file name{0}\
      - OUTPUT_SIM_ONE_FILE_PER_VARIABLE:{0}\
            one file per variable in output (flagged as 1 above),{0}\
            - requires as many file name(s) as variable(s) flagged as 1 above{0}\
      - OUTPUT_SIM_ONE_FILE_PER_REALIZATION:{0}\
           one file per realization,{0}\
           - requires one prefix (for file name, the string "_real#####.gslib"{0}\
             will be appended where "######" is the realization index){0}\
*/{0}'.format(endofline))

    infid.write('\
OUTPUT_SIM_ONE_FILE_PER_REALIZATION{0}\
{1}{0}\
{0}'.format(endofline, '{}'.format(fileprefix, suffix_outputSim)))

    # Output: additional maps
    if verbose > 0:
        infid.write('\
/* OUTPUT: ADDITIONAL MAPS */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Additional maps (images) can be retrieved in output.{0}\
   An output map is defined on the simulation grid and contains information{0}\
   on the simulation but excluding possible post-processing path(s).{0}\
   The following output maps are proposed:{0}\
      - path index map         : index in the simulation path{0}\
      - error map              : error e (see TOLERANCE below){0}\
      - TI grid node index map : index of the grid node of the retained{0}\
                                 candidate in the TI{0}\
      - TI index map           : index of the TI used (makes sense if number of{0}\
                                 TIs (nTI) is greater than 1){0}\
   For each of these 4 cases (in the order exposed above), specify:{0}\
      - flag (0 / 1) indicating if the output is desired{0}\
      - and if the flag is set to 1, a prefix for the name of the output files{0}\
   Note: for any output map, one file per realization is generated, the string{0}\
         "_real#####.gslib" will be appended to the given prefix, where "######"{0}\
         is the realization index.{0}\
*/{0}'.format(endofline))

        if deesse_input.outputPathIndexFlag:
            infid.write('{1} {2} // path index map{0}'.format(endofline, int(deesse_input.outputPathIndexFlag), '{}{}'.format(fileprefix, suffix_outputPathIndex)))
        else:
            infid.write('{1} // path index map{0}'.format(endofline, int(deesse_input.outputPathIndexFlag)))

        if deesse_input.outputErrorFlag:
            infid.write('{1} {2} // error map{0}'.format(endofline, int(deesse_input.outputErrorFlag), '{}{}'.format(fileprefix, suffix_outputError)))
        else:
            infid.write('{1} // error map{0}'.format(endofline, int(deesse_input.outputErrorFlag)))

        if deesse_input.outputTiGridNodeIndexFlag:
            infid.write('{1} {2} // TI grid node index map{0}'.format(endofline, int(deesse_input.outputTiGridNodeIndexFlag), '{}{}'.format(fileprefix, suffix_outputTiGridNodeIndex)))
        else:
            infid.write('{1} // TI grid node index map{0}'.format(endofline, int(deesse_input.outputTiGridNodeIndexFlag)))

        if deesse_input.outputTiIndexFlag:
            infid.write('{1} {2} // TI index map{0}{0}'.format(endofline, int(deesse_input.outputTiIndexFlag), '{}{}'.format(fileprefix, suffix_outputTiIndex)))
        else:
            infid.write('{1} // TI index map{0}{0}'.format(endofline, int(deesse_input.outputTiIndexFlag)))

    else:
        if deesse_input.outputPathIndexFlag:
            infid.write('{1} {2}{0}'.format(endofline, int(deesse_input.outputPathIndexFlag), '{}{}'.format(fileprefix, suffix_outputPathIndex)))
        else:
            infid.write('{1}{0}'.format(endofline, int(deesse_input.outputPathIndexFlag)))

        if deesse_input.outputErrorFlag:
            infid.write('{1} {2}{0}'.format(endofline, int(deesse_input.outputErrorFlag), '{}{}'.format(fileprefix, suffix_outputError)))
        else:
            infid.write('{1}{0}'.format(endofline, int(deesse_input.outputErrorFlag)))

        if deesse_input.outputTiGridNodeIndexFlag:
            infid.write('{1} {2}{0}'.format(endofline, int(deesse_input.outputTiGridNodeIndexFlag), '{}{}'.format(fileprefix, suffix_outputTiGridNodeIndex)))
        else:
            infid.write('{1}{0}'.format(endofline, int(deesse_input.outputTiGridNodeIndexFlag)))

        if deesse_input.outputTiIndexFlag:
            infid.write('{1} {2}{0}{0}'.format(endofline, int(deesse_input.outputTiIndexFlag), '{}{}'.format(fileprefix, suffix_outputTiIndex)))
        else:
            infid.write('{1}{0}{0}'.format(endofline, int(deesse_input.outputTiIndexFlag)))

    # Output report
    if verbose > 0:
        infid.write('\
/* OUTPUT REPORT */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Flag (0 / 1), and if 1, output report file. */{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, int(deesse_input.outputReportFlag)))
    if deesse_input.outputReportFlag:
        infid.write('{1}{0}'.format(endofline, deesse_input.outputReportFileName))
    infid.write('{0}'.format(endofline))

    # Training image
    if verbose > 0:
        infid.write('\
/* TRAINING IMAGE */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Number of training image(s) (nTI >= 1), followed by nTI file(s){0}\
   (a file can be replaced by the string "_DATA_" which means that the{0}\
   simulation grid itself is taken as training image), and{0}\
   if nTI > 1, one pdf image file (for training images, nTI variables). */{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, deesse_input.nTI))

    for i, (f, im) in enumerate(zip(deesse_input.simGridAsTiFlag, deesse_input.TI)):
        if f:
            infid.write('{1}{0}'.format(endofline, deesse.MPDS_SIMULATION_GRID_AS_TRAINING_IMAGE))
        else:
            fname = '{}{}{}.gslib'.format(fileprefix, suffix_TI, i)
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('{1}{0}'.format(endofline, fname))

    if deesse_input.nTI > 1:
        # write "pdfTI"...
        im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                 sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                 ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                 nv=deesse_input.nTI, val=deesse_input.pdfTI,
                 varname=['pdfTI{}'.format(i) for i in range(deesse_input.nTI)])
        fname = '{}{}.{}'.format(fileprefix, suffix_pdfTI, 'gslib')
        img.writeImageGslib(im, dirname + '/' + fname,
                            missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
        infid.write('{1}{0}'.format(endofline, fname))

    infid.write('{0}'.format(endofline))

    # Conditioning data files (data image file / data point set file)
    if verbose == 1:
        infid.write('\
/* CONDITIONING DATA FILES (IMAGE FILE OR POINT SET FILE) (below) */{0}{0}'.format(endofline))
    elif verbose == 2:
        infid.write('\
/* CONDITIONING DATA FILES (IMAGE FILE OR POINT SET FILE) (below){0}\
   In such files, the name of a variable should correspond to a variable name{0}\
   specified above to give usual conditioning (hard) data for that variable.{0}\
   Moreover, inequality conditioning data can be given for a variable by{0}\
   appending the suffix "_min" (resp. "_max") to the variable name: the given{0}\
   values are then minimal (resp. maximal) bounds, i.e. indicating that the{0}\
   simulated values should be greater than or equal to (resp. less than or equal{0}\
   to) the given values. */{0}{0}'.format(endofline))

    # Data image
    if verbose > 0:
        infid.write('\
/* DATA IMAGE FILE FOR SG */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Number of image file(s) (n >= 0), followed by n file(s){0}\
   (such image(s) must be defined on the simulation grid (same support)). */{0}'.format(endofline))

    if deesse_input.dataImage is not None:
        infid.write('{1}{0}'.format(endofline, len(deesse_input.dataImage)))
        for i, im in enumerate(deesse_input.dataImage):
            fname = '{}{}{}.gslib'.format(fileprefix, suffix_dataImage, i)
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('{1}{0}'.format(endofline, fname))
    else:
        infid.write('0{0}'.format(endofline))

    infid.write('{0}'.format(endofline))

    # Data point set
    if verbose > 0:
        infid.write('\
/* DATA POINT SET FILE FOR SG */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Number of point set file(s) (n >= 0), followed by n file(s). */{0}'.format(endofline))

    if deesse_input.dataPointSet is not None:
        infid.write('{1}{0}'.format(endofline, len(deesse_input.dataPointSet)))
        for i, ps in enumerate(deesse_input.dataPointSet):
            fname = '{}{}{}.gslib'.format(fileprefix, suffix_dataPointSet, i)
            img.writePointSetGslib(ps, dirname + '/' + fname,
                                   missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('{1}{0}'.format(endofline, fname))
    else:
        infid.write('0{0}'.format(endofline))

    infid.write('{0}'.format(endofline))

    # Mask
    if verbose > 0:
        infid.write('\
/* MASK IMAGE */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Flag (0: mask not used / 1: mask used) and if 1, mask image file{0}\
   (this image contains one variable on the simulation grid: flag (0 / 1){0}\
   for each node of the simulation grid that indicates if the variable(s){0}\
   will be simulated at the corresponding node (flag 1) or not (flag 0). */{0}'.format(endofline))

    if deesse_input.mask is not None:
        im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                 sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                 ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                 nv=1, val=deesse_input.mask,
                 varname='mask')
        fname = '{}{}.{}'.format(fileprefix, suffix_mask, 'gslib')
        img.writeImageGslib(im, dirname + '/' + fname,
                            missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
        infid.write('1{0}{1}{0}'.format(endofline, fname))
    else:
        infid.write('0{0}'.format(endofline))

    infid.write('{0}'.format(endofline))

    # Homothety
    if verbose > 0:
        infid.write('\
/* HOMOTHETY */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* 1. Homothety usage, integer (homothetyUsage):{0}\
        - 0: no homothety{0}\
        - 1: homothety without tolerance{0}\
        - 2: homothety with tolerance{0}\
   2a. If homothetyUsage == 1,{0}\
          then for homothety ratio in each direction,{0}\
          first for x, then for y, and then for z-axis direction:{0}\
             - Flag (0 / 1) indicating if given in an image file,{0}\
               followed by{0}\
                  - one value (real) if flag is 0{0}\
                  - name of the image file (one variable) if flag is 1{0}\
   2b. If homothetyUsage == 2,{0}\
          then for homothety ratio in each direction,{0}\
          first for x, then for y, and then for z-axis direction:{0}\
             - Flag (0 / 1) indicating if given in an image file,{0}\
               followed by{0}\
                  - two values (lower and upper bounds) (real) if flag is 0{0}\
                  - name of the image file (two variables) if flag is 1{0}\
*/{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, deesse_input.homothetyUsage))

    if deesse_input.homothetyUsage == 1:
        if deesse_input.homothetyXLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyXRatio,
                     varname='ratio')
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyXRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.homothetyXRatio[0]))

        if deesse_input.homothetyYLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyYRatio,
                     varname='ratio')
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyYRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.homothetyYRatio[0]))

        if deesse_input.homothetyZLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.homothetyZRatio,
                     varname='ratio')
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyZRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.homothetyZRatio[0]))

    elif deesse_input.homothetyUsage == 2:
        if deesse_input.homothetyXLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyXRatio,
                     varname=['ratioMin', 'ratioMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyXRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.homothetyXRatio[0], deesse_input.homothetyXRatio[1]))

        if deesse_input.homothetyYLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyYRatio,
                     varname=['ratioMin', 'ratioMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyYRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.homothetyYRatio[0], deesse_input.homothetyYRatio[1]))

        if deesse_input.homothetyZLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.homothetyZRatio,
                     varname=['ratioMin', 'ratioMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_homothetyZRatio, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.homothetyZRatio[0], deesse_input.homothetyZRatio[1]))

    infid.write('{0}'.format(endofline))

    # Rotation
    if verbose > 0:
        infid.write('\
/* ROTATION */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* 1. Rotation usage, integer (rotationUsage):{0}\
        - 0: no rotation{0}\
        - 1: rotation without tolerance{0}\
        - 2: rotation with tolerance{0}\
   2a. If rotationUsage == 1,{0}\
          then for each angle,{0}\
          first for azimuth, then for dip, and then for plunge:{0}\
             - Flag (0 / 1) indicating if given in an image file,{0}\
               followed by{0}\
                  - one value (real) if flag is 0{0}\
                  - name of the image file (one variable) if flag is 1{0}\
   2b. If rotationUsage == 2,{0}\
          then for each angle,{0}\
          first for azimuth, then for dip, and then for plunge:{0}\
             - Flag (0 / 1) indicating if given in an image file,{0}\
               followed by{0}\
                  - two values (lower and upper bounds) (real) if flag is 0{0}\
                  - name of the image file (two variables) if flag is 1{0}\
*/{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, deesse_input.rotationUsage))

    if deesse_input.rotationUsage == 1:
        if deesse_input.rotationAzimuthLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationAzimuth,
                     varname='angle')
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationAzimuth, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.rotationAzimuth[0]))

        if deesse_input.rotationDipLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationDip,
                     varname='angle')
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationDip, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.rotationDip[0]))

        if deesse_input.rotationPlungeLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=1, val=deesse_input.rotationPlunge,
                     varname='angle')
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationPlunge, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1}{0}'.format(endofline, deesse_input.rotationPlunge[0]))

    elif deesse_input.rotationUsage == 2:
        if deesse_input.rotationAzimuthLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationAzimuth,
                     varname=['angleMin', 'angleMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationAzimuth, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.rotationAzimuth[0], deesse_input.rotationAzimuth[1]))

        if deesse_input.rotationDipLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationDip,
                     varname=['angleMin', 'angleMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationDip, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.rotationDip[0], deesse_input.rotationDip[1]))

        if deesse_input.rotationPlungeLocal:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=2, val=deesse_input.rotationPlunge,
                     varname=['angleMin', 'angleMax'])
            fname = '{}{}.{}'.format(fileprefix, suffix_rotationPlunge, 'gslib')
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
            infid.write('1 {1}{0}'.format(endofline, fname))

        else:
            infid.write('0 {1} {2}{0}'.format(endofline, deesse_input.rotationPlunge[0], deesse_input.rotationPlunge[1]))

    infid.write('{0}'.format(endofline))

    # Consistency of conditioning data
    if verbose > 0:
        infid.write('\
/* CONSISTENCY OF CONDITIONING DATA (TOLERANCE RELATIVELY TO THE RANGE OF TRAINING VALUES) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Maximal expansion (expMax): real number (negative to not check consistency).{0}\
   The following is applied for each variable separetely:{0}\
      - For variable with distance type set to 0 (see below):{0}\
           * expMax >= 0:{0}\
                if a conditioning data value is not in the set of training image values,{0}\
                an error occurs{0}\
           * expMax < 0:{0}\
                if a conditioning data value is not in the set of training image values,{0}\
                a warning is displayed (no error occurs){0}\
      - For variable with distance type not set to 0 (see below): if relative distance{0}\
        flag is set to 1 (see below), nothing is done, else:{0}\
           * expMax >= 0: maximal accepted expansion of the range of the training image values{0}\
                for covering the conditioning data values:{0}\
                - if conditioning data values are within the range of the training image values:{0}\
                     nothing is done{0}\
                - if a conditioning data value is out of the range of the training image values:{0}\
                     let{0}\
                        new_min_ti = min ( min_cd, min_ti ){0}\
                        new_max_ti = max ( max_cd, max_ti ){0}\
                     with{0}\
                        min_cd, max_cd, the min and max of the conditioning values,{0}\
                        min_ti, max_ti, the min and max of the training imges values.{0}\
                     If new_max_ti - new_min_ti <= (1 + expMax) * (ti_max - ti_min), then{0}\
                     the training image values are linearly rescaled from [ti_min, ti_max] to{0}\
                     [new_ti_min, new_ti_max], and a warning is displayed (no error occurs).{0}\
                     Otherwise, an error occurs.{0}\
           * expMax < 0: if a conditioning data value is out of the range of the training image{0}\
                values, a warning is displayed (no error occurs), the training image values are{0}\
                not modified.{0}\
*/{0}'.format(endofline))

    infid.write('{1}{0}{0}'.format(endofline, deesse_input.expMax))

    # Normalization
    if verbose > 0:
        infid.write('\
/* NORMALIZATION TYPE (FOR VARIABLES FOR WHICH DISTANCE TYPE IS NOT 0 AND DISTANCE IS ABSOLUTE) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Available types:{0}\
      - NORMALIZING_LINEAR{0}\
      - NORMALIZING_UNIFORM{0}\
      - NORMALIZING_NORMAL */{0}'.format(endofline))

    if deesse_input.normalizingType == 'linear':
        infid.write('NORMALIZING_LINEAR{0}{0}'.format(endofline))
    elif deesse_input.normalizingType == 'uniform':
        infid.write('NORMALIZING_UNIFORM{0}{0}'.format(endofline))
    elif deesse_input.normalizingType == 'normal':
        infid.write('NORMALIZING_NORMAL{0}{0}'.format(endofline))

    # Search neighborhood parameters
    if verbose > 0:
        infid.write('\
/* SEARCH NEIGHBORHOOD PARAMETERS */{0}'.format(endofline))

    if verbose == 2:
        infid.write("\
/* A search neighborhood is a 3D ellipsoid, defined by:{0}\
      - (rx, ry, rz): search radius (in number of nodes) for each direction{0}\
      - (ax, ay, az): anisotropy ratio or inverse distance unit for each direction;{0}\
        the distance to the central node is computed as the Euclidean distance by{0}\
        considering the nodes with the unit 1/ax x 1/ay x 1/az;{0}\
      - (angle1, angle2, angle3): angles (azimuth, dip and plunge) defining the{0}\
        rotation that sends the coordinates system xyz onto the coordinates{0}\
        system x'y'z' in which the search radii and the anisotropy ratios are given.{0}\
      - power: at which the distance is elevated for computing the weight of each{0}\
        node in the search neighborhood.{0}\
   Note that{0}\
     - the search neighborhood is delimited by the search radii and the angles{0}\
     - the anisotropy ratios are used only for computing the distance to the central{0}\
       node, from each node in the search neighborhood{0}\
     - the nodes inside the search neighborhood are sorted according to their{0}\
       distance to the central node, from the closest one to the furthest one{0}\
{0}\
   FOR EACH VARIABLE:{0}\
      1. search radii, available specifications (format: key word [parameters]):{0}\
            SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ{0}\
            SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ{0}\
            SEARCHNEIGHBORHOOD_RADIUS_MANUAL rx ry rz{0}\
         with the following meaning, where tx, ty, tz denote the ranges of the{0}\
         variogram of the TI(s) in x-, y-, z-directions:{0}\
            - SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT:{0}\
                 large radii set according to the size of the SG and the TI(s),{0}\
                 and the use of homothethy and/or rotation for the simulation{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT:{0}\
                 search radii set according to the TI(s) variogram ranges,{0}\
                 one of the 5 next modes SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE* will be{0}\
                 used according to the use of homothethy and/or rotation for the{0}\
                 simulation{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE:{0}\
                 rx prop. to tx, ry prop. to ty, rz prop. to tz,{0}\
                 independently in each direction{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY:{0}\
                 rx = ry  prop. to max(tx, ty), independently from rz prop. to tz{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ:{0}\
                 rx = rz  prop. to max(tx, tz), independently from ry prop. to ty{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ:{0}\
                 ry = rz  prop. to max(ty, tz), independently from rx prop. to tx{0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ:{0}\
                 rx = ry = rz prop. to max(tx, ty, tz){0}\
                 (automatically computed){0}\
            - SEARCHNEIGHBORHOOD_RADIUS_MANUAL:{0}\
                 search radii are explicitly given{0}\
{0}\
      2. anisotropy ratios, available specifications (format: key word [parameters]):{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ{0}\
            SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_MANUAL ax ay az{0}\
         with the following meaning:{0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE:{0}\
                 ax = ay = az = 1{0}\
                 (isotropic distance:{0}\
                 maximal distance for search neighborhood nodes will be equal to the{0}\
                 maximum of the search radii){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS:{0}\
                 ax = rx, ay = ry, az = rz{0}\
                 (anisotropic distance:{0}\
                 nodes at distance one on the border of the search neighborhood,{0}\
                 maximal distance for search neighborhood nodes will be 1){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY:{0}\
                 ax = ay = max(rx, ry), az = rz{0}\
                 (anisotropic distance:{0}\
                 maximal distance for search neighborhood nodes will be 1){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ:{0}\
                 ax = az = max(rx, rz), ay = ry{0}\
                 (anisotropic distance:{0}\
                 maximal distance for search neighborhood nodes will be 1){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ:{0}\
                 ay = az = max(ry, rz), ax = rx{0}\
                 (anisotropic distance:{0}\
                 maximal distance for search neighborhood nodes will be 1){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ:{0}\
                 ax = ay = az = max(rx, ry, rz){0}\
                 (isotropic distance:{0}\
                 maximal distance for search neighborhood nodes will be 1){0}\
            - SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_MANUAL:{0}\
                 anisotropy ratios are explicitly given{0}\
{0}\
      3. rotation, available specifications (format: key word [parameters]):{0}\
            SEARCHNEIGHBORHOOD_ROTATION_OFF{0}\
            SEARCHNEIGHBORHOOD_ROTATION_ON angle1 angle2 angle3{0}\
         with the following meaning:{0}\
            - SEARCHNEIGHBORHOOD_ROTATION_OFF:{0}\
                 search neighborhood is not rotated{0}\
            - SEARCHNEIGHBORHOOD_ROTATION_ON:{0}\
                 search neighborhood is rotated, the rotation is described by{0}\
                 angle1 (azimuth), angle2 (dip) and angle3 (plunge){0}\
{0}\
      4. power (real number){0}\
*/{0}".format(endofline))

    for i, sn in enumerate(deesse_input.searchNeighborhoodParameters):
        if verbose > 0:
            infid.write('\
/* SEARCH NEIGHBORHOOD PARAMETERS FOR VARIABLE #{1} */{0}'.format(endofline, i))

        if sn.radiusMode == 'large_default':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT')
        elif sn.radiusMode == 'ti_range_default':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT')
        elif sn.radiusMode == 'ti_range':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE')
        elif sn.radiusMode == 'ti_range_xy':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY')
        elif sn.radiusMode == 'ti_range_xz':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ')
        elif sn.radiusMode == 'ti_range_yz':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ')
        elif sn.radiusMode == 'ti_range_xyz':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ')
        elif sn.radiusMode == 'manual':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_MANUAL {0} {1} {2}'.format(sn.rx, sn.ry, sn.rz))
        if verbose == 2:
            infid.write(' // search radii{0}'.format(endofline))
        else:
            infid.write('{0}'.format(endofline))

        if sn.anisotropyRatioMode == 'one':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE')
        elif sn.anisotropyRatioMode == 'radius':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS')
        elif sn.anisotropyRatioMode == 'radius_xy':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY')
        elif sn.anisotropyRatioMode == 'radius_xz':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ')
        elif sn.anisotropyRatioMode == 'radius_yz':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ')
        elif sn.anisotropyRatioMode == 'radius_xyz':
            infid.write('SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ')
        elif sn.anisotropyRatioMode == 'manual':
            infid.write('SEARCHNEIGHBORHOOD_RADIUS_MANUAL {0} {1} {2}'.format(sn.ax, sn.ay, sn.az))
        if verbose == 2:
            infid.write(' // anisotropy ratios{0}'.format(endofline))
        else:
            infid.write('{0}'.format(endofline))

        if sn.angle1 == 0 and sn.angle2 == 0 and sn.angle3 == 0:
            infid.write('SEARCHNEIGHBORHOOD_ROTATION_OFF')
        else:
            infid.write('SEARCHNEIGHBORHOOD_ROTATION_ON {0} {1} {2}'.format(sn.angle1, sn.angle2, sn.angle3))
        if verbose == 2:
            infid.write(' // rotation{0}'.format(endofline))
        else:
            infid.write('{0}'.format(endofline))

        if verbose == 2:
            infid.write('{1} // power for computing weight according to distance{0}'.format(endofline, sn.power))
        else:
            infid.write('{1}{0}'.format(endofline, sn.power))

        infid.write('{0}'.format(endofline))

    # Maximal number of neighbors
    if verbose > 0:
        infid.write('\
/* MAXIMAL NUMBER OF NEIGHBORING NODES FOR EACH VARIABLE (as many number(s) as number of variable(s)) */{0}'.format(endofline))

    for v in deesse_input.nneighboringNode:
        infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Maximal proportion of nodes with inequality data in pattern if the maximal number of nodes is reached
    if verbose > 0:
        infid.write('\
/* MAXIMAL PROPORTION OF NEIGHBORING NODES WITH INEQUALITY DATA (WHEN THE MAXIMAL NUMBER OF NEIGHBORING{0}\
   NODES IS REACHED) FOR EACH VARIABLE (as many number(s) as number of variable(s)) */{0}'.format(endofline))

    for v in deesse_input.maxPropInequalityNode:
        infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Maximal density of neighbors
    if verbose > 0:
        infid.write('\
/* MAXIMAL DENSITY OF NEIGHBORING NODES IN SEARCH NEIGHBORHOOD FOR EACH VARIABLE (as many number(s){0}\
   as number of variable(s)) */{0}'.format(endofline))

    for v in deesse_input.neighboringNodeDensity:
        infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Rescaling mode
    if verbose > 0:
        infid.write('\
/* RESCALING MODE FOR EACH VARIABLE (as many specification as number of variable(s)) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Available specifications (format: key word [parameters]):{0}\
      RESCALING_NONE{0}\
      RESCALING_MIN_MAX min_value max_value{0}\
      RESCALING_MEAN_LENGTH mean_value length_value{0}\
   with the following meaning:{0}\
      - RESCALING_NONE:{0}\
           target interval for simulated values = interval of training image(s) value(s){0}\
           (standard mode){0}\
      - RESCALING_MIN_MAX:{0}\
           target interval for simulated values given by min_value and max_value{0}\
      - RESCALING_MEAN_LENGTH:{0}\
           target interval for simulated values given by mean_value and length_value{0}\
   For the two last modes, a linear correspondence is set between the target interval{0}\
   for simulated values and the interval of the training image(s):{0}\
      - for mode RESCALING_MIN_MAX, min_value and max_value are set in correspondence{0}\
        with the min and max training image(s) values respectively{0}\
      - for mode RESCALING_MEAN_LENGTH, mean_value is set in correspondence{0}\
        with the mean training image(s) value, and length_value is the length{0}\
        of the target interval.{0}\
   Note that:{0}\
      - conditioning data values (if any) must be in the target interval.{0}\
      - the two last modes are not compatible with distance type 0{0}\
        (non-matching nodes), or with relative distance mode (see below){0}\
      - class of values (for other options such probability constraints or{0}\
        connectivity constraints) are given according to the target interval.{0}\
*/{0}'.format(endofline))

    for i, m in enumerate(deesse_input.rescalingMode):
        if m == 'none':
            infid.write('RESCALING_NONE{0}'.format(endofline))
        elif m == 'min_max':
            infid.write('RESCALING_MIN_MAX {1} {2} {0}'.format(endofline, deesse_input.rescalingTargetMin[i], deesse_input.rescalingTargetMax[i]))
        elif m == 'mean_length':
            infid.write('RESCALING_MEAN_LENGTH {1} {2} {0}'.format(endofline, deesse_input.rescalingTargetMean[i], deesse_input.rescalingTargetLength[i]))

    infid.write('{0}'.format(endofline))

    # Relative distance flag
    if verbose > 0:
        infid.write('\
/* RELATIVE DISTANCE FLAG FOR EACH VARIABLE (as many flag(s) (0 / 1) as number of variable(s)) */{0}'.format(endofline))

    for flag in deesse_input.relativeDistanceFlag:
        infid.write('{1}{0}'.format(endofline, int(flag)))
    infid.write('{0}'.format(endofline))

    # Distance type
    if verbose > 0:
        infid.write('\
/* DISTANCE TYPE FOR EACH VARIABLE (as many number(s) as number of variable(s)) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Available distance (between data events):{0}\
      - 0: non-matching nodes (typically for categorical variable){0}\
      - 1: L-1 distance{0}\
      - 2: L-2 distance{0}\
      - 3: L-p distance, requires the real positive parameter p{0}\
      - 4: L-infinity distance */{0}'.format(endofline))

    for i, v in enumerate(deesse_input.distanceType):
        if v == 3:
            infid.write('{1} {2}{0}'.format(endofline, v, deesse_input.powerLpDistance[i]))
        else:
            infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Weight factor for conditioning data
    if verbose > 0:
        infid.write('\
/* WEIGHT FACTOR FOR CONDITIONING DATA, FOR EACH VARIABLE (as many number(s) as number of variable(s)) */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* For the computation of distance between data events, if a value is a conditioning{0}\
   data, its corresponding contribution is multiplied by the factor given here. */{0}'.format(endofline))

    for v in deesse_input.conditioningWeightFactor:
        infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Simulation type
    if verbose > 0:
        infid.write('\
/* SIMULATION TYPE */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* Key word:{0}\
      - SIM_ONE_BY_ONE:{0}\
           successive simulation of one variable at one node in the simulation grid (4D path){0}\
      - SIM_VARIABLE_VECTOR:{0}\
           successive simulation of all variable(s) at one node in the simulation grid (3D path) */{0}'.format(endofline))

    if deesse_input.simType == 'sim_one_by_one':
        infid.write('SIM_ONE_BY_ONE{0}{0}'.format(endofline))
    elif deesse_input.simType == 'sim_variable_vector':
        infid.write('SIM_VARIABLE_VECTOR{0}{0}'.format(endofline))

    # Simulation path
    if verbose > 0:
        infid.write('\
/* SIMULATION PATH */{0}'.format(endofline))

    if verbose == 2:
        infid.write("\
/* Available paths (format: key word [parameters]):{0}\
      - PATH_RANDOM{0}\
      - PATH_RANDOM_HD_DISTANCE_PDF s{0}\
      - PATH_RANDOM_HD_DISTANCE_SORT s{0}\
      - PATH_RANDOM_HD_DISTANCE_SUM_PDF p s{0}\
      - PATH_RANDOM_HD_DISTANCE_SUM_SORT p s{0}\
      - PATH_UNILATERAL u (vector){0}\
   with the following meaning:{0}\
      - PATH_RANDOM:{0}\
           Random path, for simulation type:{0}\
              - SIM_ONE_BY_ONE     : path visiting all nodes and variables in a random order{0}\
              - SIM_VARIABLE_VECTOR: path visiting all nodes in a random order{0}\
      - PATH_RANDOM_HD_DISTANCE_PDF:{0}\
           This path preferentially visits first the SG nodes close to the conditioning nodes.{0}\
           For any uninformed node, the distance d to the set of conditioning nodes is computed,{0}\
           and a weight proportional to a^d is attached, with 0 < a = 1 + s*(a0-1) <= 1, a0{0}\
           being the minimal value for a.{0}\
           The path is then built by successively drawing nodes according to their weight.{0}\
           The required parameter s in [0,1] controls the strength of the distance{0}\
           (s = 0: random, s = 1: most guided by the distance).{0}\
           Notes: 1) in absence of conditioning node, one random node in the path is considered{0}\
           to compute the distances, 2) if simulation type is SIM_VARIABLE_VECTOR, variables{0}\
           exhaustively informed are not considered as conditioning data for computing distance.{0}\
     - PATH_RANDOM_HD_DISTANCE_SORT:{0}\
           This path preferentially visits first the SG nodes close to the conditioning nodes.{0}\
           For any uninformed node, the distance d to the set of conditioning nodes is computed,{0}\
           and the normalized distance d' in [0,1] is combined with a random number r in [0,1]{0}\
           to define a weight w = s*d' + (1-s)*r.{0}\
           The path is then built by sorting the nodes such that the weight are in increasing order.{0}\
           The required parameter s in [0,1] controls the strength of the distance{0}\
           (s = 0: random, s = 1: most guided by the distance - quasi deterministic).{0}\
           Notes: 1) in absence of conditioning node, one random node in the path is considered{0}\
           to compute the distances, 2) if simulation type is SIM_VARIABLE_VECTOR, variables{0}\
           exhaustively informed are not considered as conditioning data for computing distance.{0}\
      - PATH_RANDOM_HD_DISTANCE_SUM_PDF:{0}\
           The path is built as for PATH_RANDOM_HD_DISTANCE_PDF, but the distance to the set of{0}\
           conditioning nodes is replaced by the sum of the distances to every conditioning node,{0}\
           each distance being raised to a power p.{0}\
           Two parameters are required: p > 0 controls the computation of the distances,{0}\
           and s in [0,1] controls the strength of the distance{0}\
           (s = 0: random, s = 1: most guided by the distance).{0}\
           Note: 1) in absence of conditioning node, random path is considered.{0}\
      - PATH_RANDOM_HD_DISTANCE_SUM_SORT:{0}\
           The path is built as for PATH_RANDOM_HD_DISTANCE_SORT, but the distance to the set of{0}\
           conditioning nodes is replaced by the sum of the distances to every conditioning node,{0}\
           each distance being raised to a power p.{0}\
           Two parameters are required: p > 0 controls the computation of the distances,{0}\
           and s in [0,1] controls the strength of the distance{0}\
           (s = 0: random, s = 1: most guided by the distance - quasi deterministic).{0}\
           Note: 1) in absence of conditioning node, random path is considered.{0}\
      - PATH_UNILATERAL:{0}\
           It requires a vector u of parameters of size n=4 (resp. n=3) for simulation type set to{0}\
           SIM_ONE_BY_ONE (resp. SIM_VARIABLE_VECTOR), given as ux uy uz uv (resp. ux uy uz).{0}\
           Some components of u can be equal to 0, and the other ones must be the integer 1,...,m,{0}\
           with sign +/-. For{0}\
               u[j(1)] = ... = u[j(n-m)] = 0 and (u[i(1)], ..., u[i(m)]) = (+/-1,...,+/-m),{0}\
           the path visits all nodes in sections of coordinates j(1),...,j(n-m) randomly,{0}\
           and makes varying the i(1)-th coordinate first (most rapidly), ..., the i(m)-th coordinate{0}\
           last, each one in increasing or decreasing order according to the sign (+ or - resp.).{0}\
           Examples:{0}\
             - for simulation type SIM_ONE_BY_ONE, and u = (0, -2, 1, 0), then the path will visit{0}\
               all nodes: randomly in xv-sections, with increasing z-coordinate first (most rapidly),{0}\
               and decreasing y-coordinate.{0}\
             - for simulation type SIM_VARIABLE_VECTOR, and u = (0, 2, 1), then the path will visit{0}\
               all nodes: randomly in x-direction, with increasing z-coordinate first (most rapidly),{0}\
               and increasing y-coordinate.{0}\
*/{0}".format(endofline))

    if deesse_input.simPathType == 'random':
        infid.write('PATH_RANDOM{0}'.format(endofline))
    elif deesse_input.simPathType == 'random_hd_distance_pdf':
        infid.write('PATH_RANDOM_HD_DISTANCE_PDF {1}{0}'.format(endofline, deesse_input.simPathStrength))
    elif deesse_input.simPathType == 'random_hd_distance_sort':
        infid.write('PATH_RANDOM_HD_DISTANCE_SORT {1}{0}'.format(endofline, deesse_input.simPathStrength))
    elif deesse_input.simPathType == 'random_hd_distance_sum_pdf':
        infid.write('PATH_RANDOM_HD_DISTANCE_SUM_PDF {1} {2}{0}'.format(endofline, deesse_input.simPathPower, deesse_input.simPathStrength))
    elif deesse_input.simPathType == 'random_hd_distance_sum_sort':
        infid.write('PATH_RANDOM_HD_DISTANCE_SUM_SORT {1} {2}{0}'.format(endofline, deesse_input.simPathPower, deesse_input.simPathStrength))
    elif deesse_input.simPathType == 'unilateral':
        infid.write('PATH_UNILATERAL')
        for n in deesse_input.simPathUnilateralOrder:
            infid.write(' {}'.format(n))
        infid.write('{0}'.format(endofline))

    infid.write('{0}'.format(endofline))

    # Distance threshold
    if verbose > 0:
        infid.write('\
/* DISTANCE THRESHOLD FOR EACH VARIABLE (as many number(s) as number of variable(s)) */{0}'.format(endofline))

    for v in deesse_input.distanceThreshold:
        infid.write('{1}{0}'.format(endofline, v))
    infid.write('{0}'.format(endofline))

    # Probability constraints
    if verbose > 0:
        infid.write('\
/* PROBABILITY CONSTRAINTS */{0}'.format(endofline))

    if verbose == 2:
        infid.write("\
/* FOR EACH VARIABLE:{0}\
   1. Probability constraint usage, integer (probabilityConstraintUsage):{0}\
        - 0: no probability constraint{0}\
        - 1: global probability constraint{0}\
        - 2: local probability constraint{0}\
{0}\
   2. If probabilityConstraintUsage > 0, then the classes of values (for which the{0}\
         probability constraints will be given) have to be defined; a class of values{0}\
         is given by a union of interval(s): [inf_1,sup_1[ U ... U [inf_n,sup_n[;{0}\
         Here are given:{0}\
            - nclass: number of classes of values{0}\
            - for i in 1,..., nclass: definition of the i-th class of values:{0}\
                 - ninterval: number of interval(s){0}\
                 - inf_1 sup_1 ... inf_ninterval sup_ninterval: inf and sup for each interval{0}\
                      these values should satisfy inf_i < sup_i{0}\
{0}\
   3a. If probabilityConstraintUsage == 1, then{0}\
          - global probability for each class (defined in 2. above), i.e.{0}\
            nclass numbers in [0,1] of sum 1{0}\
   3b. If probabilityConstraintUsage == 2, then{0}\
          - one pdf image file (for every class, nclass variables){0}\
            (image of same dimensions as the simulation grid){0}\
          - support radius for probability maps (i.e. distance according to{0}\
            the unit defined in the search neighborhood parameters for the{0}\
            considered variable){0}\
          - method for computing the current pdf (in the simulation grid),{0}\
             integer (localCurrentPdfComputation):{0}\
               - 0: \"COMPLETE\" mode: all the informed node in the search neighborhood{0}\
                    for the considered variable, and within the support are taken into account{0}\
               - 1: \"APPROXIMATE\" mode: only the neighboring nodes (used for the{0}\
                    search in the TI) within the support are taken into account{0}\
{0}\
   4. If probabilityConstraintUsage > 0, then{0}\
         method for comparing pdf's, integer (comparingPdfMethod):{0}\
            - 0: MAE (Mean Absolute Error){0}\
            - 1: RMSE (Root Mean Squared Error){0}\
            - 2: KLD (Kullback Leibler Divergence){0}\
            - 3: JSD (JSD (Jensen-Shannon Divergence){0}\
            - 4: MLikRsym (Mean Likelihood Ratio (over each class indicator, symmetric target interval)){0}\
            - 5: MLikRopt (Mean Likelihood Ratio (over each class indicator, optimal target interval)){0}\
{0}\
   5. If probabilityConstraintUsage > 0, then{0}\
         - deactivation distance, i.e. one positive number{0}\
           (the probability constraint is deactivated if the distance between{0}\
           the current simulated node and the last node in its neighbors (used{0}\
           for the search in the TI) (distance computed according to the corresponding{0}\
           search neighborhood parameters) is below the given deactivation distance){0}\
{0}\
   6. If probabilityConstraintUsage > 0, then{0}\
         - threshold type for pdf's comparison, integer (probabilityConstraintThresholdType){0}\
              - 0: constant threshold{0}\
              - 1: dynamic threshold{0}\
           note: if comparingPdfMethod is set to 4 or 5, probabilityConstraintThresholdType must be set to 0{0}\
         6.1a If probabilityConstraintThresholdType == 0, then{0}\
                 - threshold value{0}\
         6.1b If probabilityConstraintThresholdType == 1, then the 7 parameters:{0}\
                 - M1 M2 M3{0}\
                 - T1 T2 T3{0}\
                 - W{0}\
              These parameters should satisfy:{0}\
                 0 <= M1 <= M2 < M3,{0}\
                 T1 >= T2 >= T3,{0}\
                 w != 0.{0}\
              The threshold value t is defined as a function of the number M{0}\
              of nodes used for computing the current pdf (in the simulation grid){0}\
              including the candidate (i.e. current simulated) node by:{0}\
                 t(M) = T1, if M < M1{0}\
                 t(M) = T2, if M1 <= M < M2{0}\
                 t(M) = (T3 - T2) / (M3^W - M2^W) * (M^W - M2^W) + T2, if M2 <= M < M3{0}\
                 t(M) = T3, if M3 <= M{0}\
*/{0}".format(endofline))

    for i, sp in enumerate(deesse_input.softProbability):
        if verbose > 0:
            infid.write('\
/* PROBABILITY CONSTRAINTS FOR VARIABLE #{1} */{0}'.format(endofline, i))

        if sp.probabilityConstraintUsage == 0:
            infid.write('{1}{0}'.format(endofline, sp.probabilityConstraintUsage))
            continue

        elif sp.probabilityConstraintUsage == 1:
            if verbose == 2:
                infid.write('{1} // global probability constraint{0}'.format(endofline, sp.probabilityConstraintUsage))
            else:
                infid.write('{1}{0}'.format(endofline, sp.probabilityConstraintUsage))

        elif sp.probabilityConstraintUsage == 2:
            if verbose == 2:
                infid.write('{1} // local probability constraint{0}'.format(endofline, sp.probabilityConstraintUsage))
            else:
                infid.write('{1}{0}'.format(endofline, sp.probabilityConstraintUsage))

        if verbose == 2:
            infid.write('{1} // nclass{0}'.format(endofline, sp.nclass))
        else:
            infid.write('{1}{0}'.format(endofline, sp.nclass))

        for j, ci in enumerate(sp.classInterval):
            infid.write('{}  '.format(len(ci)))
            for inter in ci:
                infid.write(' {} {}'.format(inter[0], inter[1]))

            if verbose == 2:
                infid.write(' // class #{1} (ninterval, and interval(s)){0}'.format(endofline, j))
            else:
                infid.write('{0}'.format(endofline, j))

        if sp.probabilityConstraintUsage == 1:
            for v in sp.globalPdf:
                infid.write(' {}'.format(v))

            if verbose == 2:
                infid.write(' // global pdf{0}'.format(endofline))
            else:
                infid.write('{0}'.format(endofline))

        elif sp.probabilityConstraintUsage == 2:
            im = Img(nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                     sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                     ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                     nv=sp.nclass, val=sp.localPdf,
                     varname=['pdfClass{}'.format(j) for j in range(sp.nclass)])
            fname = '{}{}{}.gslib'.format(fileprefix, suffix_localPdf, i)
            img.writeImageGslib(im, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")

            if verbose == 2:
                infid.write('{1} // local pdf file{0}'.format(endofline, fname))
                infid.write('{1} // support radius{0}'.format(endofline, sp.localPdfSupportRadius))
                infid.write('{1} // computing local current pdf mode{0}'.format(endofline, sp.localCurrentPdfComputation))
            else:
                infid.write('{1}{0}'.format(endofline, fname))
                infid.write('{1}{0}'.format(endofline, sp.localPdfSupportRadius))
                infid.write('{1}{0}'.format(endofline, sp.localCurrentPdfComputation))

        if verbose == 2:
            infid.write('{1} // comparing pdf method{0}'.format(endofline, sp.comparingPdfMethod))
            infid.write('{1} // deactivation distance{0}'.format(endofline, sp.deactivationDistance))
            infid.write('{1} // threshold type{0}'.format(endofline, sp.probabilityConstraintThresholdType))
        else:
            infid.write('{1}{0}'.format(endofline, sp.comparingPdfMethod))
            infid.write('{1}{0}'.format(endofline, sp.deactivationDistance))
            infid.write('{1}{0}'.format(endofline, sp.probabilityConstraintThresholdType))

        if sp.probabilityConstraintThresholdType == 0:
            if verbose == 2:
                infid.write('{1} // (constant) threshold{0}'.format(endofline, sp.constantThreshold))
            else:
                infid.write('{1}{0}'.format(endofline, sp.constantThreshold))

        elif sp.probabilityConstraintThresholdType == 1:
            for v in sp.dynamicThresholdParameters:
                infid.write(' {}'.format(v))

            if verbose == 2:
                infid.write(' // dynamic threshold parameters{0}'.format(endofline))
            else:
                infid.write('{0}'.format(endofline))

    infid.write('{0}'.format(endofline))

    # Connectivity constraints
    if verbose > 0:
        infid.write('\
/* CONNECTIVITY CONSTRAINTS */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* FOR EACH VARIABLE:{0}\
   1. Connectivity constraint usage, integer (connectivityConstraintUsage):{0}\
         - 0: no connectivity constraint{0}\
         - 1: set connecting paths before simulation by successively{0}\
              binding the nodes to be connected in a random order{0}\
         - 2: set connecting paths before simulation by successively{0}\
              binding the nodes to be connected beginning with{0}\
              the pair of nodes with the smallest distance and then{0}\
              the remaining nodes in increasing order according to{0}\
              their distance to the set of nodes already connected.{0}\
         - 3: check connectivity pattern during simulation{0}\
   2. If connectivityConstraintUsage > 0, then key word for type of connectivity:{0}\
         - CONNECT_FACE             : 6-neighbors connected (by face){0}\
         - CONNECT_FACE_EDGE        : 18-neighbors connected (by face or edge){0}\
         - CONNECT_FACE_EDGE_CORNER : 26-neighbors connected (by face, edge or corner){0}\
   3. If connectivityConstraintUsage > 0, then the classes of values (that can{0}\
         be considered in the same connected components) have to be defined; a{0}\
         class of values is given by a union of interval(s):{0}\
            [inf_1,sup_1[ U ... U [inf_n,sup_n[;{0}\
         Here are given:{0}\
            - nclass: number of classes of values{0}\
            - for i in 1,..., nclass: definition of the i-th class of values:{0}\
                 - ninterval: number of interval(s){0}\
                 - inf_1 sup_1 ... inf_ninterval sup_ninterval: inf and sup for each interval{0}\
                      these values should satisfy inf_i < sup_i{0}\
   4. If connectivityConstraintUsage > 0, then:{0}\
         - variable name for connected component label{0}\
           (included in data image / data point set above){0}\
           (Note: label negative or zero means no connectivity constraint){0}\
   5a. If connectivityConstraintUsage == 1 or connectivityConstraintUsage == 2, then:{0}\
         - name of the image file and name of the variable for the search of{0}\
           connected paths (set string "_TI_" instead for searching in the (first){0}\
           training image and the corresponding variable index){0}\
   5b. If connectivityConstraintUsage == 3, then:{0}\
         - deactivation distance, i.e. one positive number{0}\
           (the connectivity constraint is deactivated if the distance between{0}\
           the current simulated node and the last node in its neighbors (used{0}\
           for the search in the TI) (distance computed according to the corresponding{0}\
           search neighborhood parameters) is below the given deactivation distance){0}\
         - threshold: threshold value for acceptation of connectivity pattern{0}\
*/{0}'.format(endofline))

    for i, co in enumerate(deesse_input.connectivity):
        if verbose > 0:
            infid.write('\
/* CONNECTIVITY CONSTRAINTS FOR VARIABLE #{1} */{0}'.format(endofline, i))

        if co.connectivityConstraintUsage == 0:
            infid.write('{1}{0}'.format(endofline, co.connectivityConstraintUsage))
            continue

        elif co.connectivityConstraintUsage == 1:
            if verbose == 2:
                infid.write('{1} // pasting connecting paths (before simulation) (random order){0}'.format(endofline, co.connectivityConstraintUsage))
            else:
                infid.write('{1}{0}'.format(endofline, co.connectivityConstraintUsage))

        elif co.connectivityConstraintUsage == 2:
            if verbose == 2:
                infid.write('{1} // pasting connecting paths (before simulation) (order according to distance){0}'.format(endofline, co.connectivityConstraintUsage))
            else:
                infid.write('{1}{0}'.format(endofline, co.connectivityConstraintUsage))

        elif co.connectivityConstraintUsage == 3:
            if verbose == 2:
                infid.write('{1} // connectivity set during simulation (patterns of labels){0}'.format(endofline, co.connectivityConstraintUsage))
            else:
                infid.write('{1}{0}'.format(endofline, co.connectivityConstraintUsage))

        if co.connectivityType == 'connect_face':
            infid.write('CONNECT_FACE'.format(endofline))
        elif co.connectivityType == 'connect_face_edge':
            infid.write('CONNECT_FACE_EDGE'.format(endofline))
        elif co.connectivityType == 'connect_face_edge_corner':
            infid.write('CONNECT_FACE_EDGE_CORNER'.format(endofline))

        if verbose == 2:
            infid.write(' // connectivity type{0}'.format(endofline))
        else:
            infid.write('{0}'.format(endofline))

        if verbose == 2:
            infid.write('{1} // nclass{0}'.format(endofline, co.nclass))
        else:
            infid.write('{1}{0}'.format(endofline, co.nclass))

        for j, ci in enumerate(co.classInterval):
            infid.write('{}  '.format(len(ci)))
            for inter in ci:
                infid.write(' {} {}'.format(inter[0], inter[1]))

            if verbose == 2:
                infid.write(' // class #{1} (ninterval, and interval(s)){0}'.format(endofline, j))
            else:
                infid.write('{0}'.format(endofline, j))

        if verbose == 2:
            infid.write('{1} // name for connected component label{0}'.format(endofline, co.varname))
        else:
            infid.write('{1}{0}'.format(endofline, co.varname))

        if co.connectivityConstraintUsage == 1 or co.connectivityConstraintUsage == 2:
            if co.tiAsRefFlag:
                infid.write('{}'.format(deesse.MPDS_USE_TRAINING_IMAGE_FOR_CONNECTIVITY))
            else:
                fname = '{}{}{}.gslib'.format(fileprefix, suffix_refConnectivityImage, i)
                img.writeImageGslib(co.refConnectivityImage, dirname + '/' + fname,
                                missing_value=deesse.MPDS_MISSING_VALUE, fmt="%.10g")
                infid.write('{0} {1}'.format(fname, co.refConnectivityImage.varname[co.refConnectivityVarIndex]))

            if verbose == 2:
                infid.write(' // reference image (and variable){0}'.format(endofline))
            else:
                infid.write('{0}'.format(endofline))

        elif co.connectivityConstraintUsage == 3:
            if verbose == 2:
                infid.write('{1} // deactivation distance{0}'.format(endofline, co.deactivationDistance))
                infid.write('{1} // threshold{0}'.format(endofline, co.threshold))
            else:
                infid.write('{1}{0}'.format(endofline, co.deactivationDistance))
                infid.write('{1}{0}'.format(endofline, co.threshold))

    infid.write('{0}'.format(endofline))

    # Block data
    if verbose > 0:
        infid.write('\
/* BLOCK DATA */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* FOR EACH VARIABLE:{0}\
   1. Block data usage, integer (blockDataUsage):{0}\
         - 0: no block data{0}\
         - 1: use of block data (mean value){0}\
{0}\
   2. If blockDataUsage == 1, then{0}\
         - block data file name{0}\
*/{0}'.format(endofline))

    for i, bd in enumerate(deesse_input.blockData):
        if verbose > 0:
            infid.write('\
/* BLOCK DATA FOR VARIABLE #{1} */{0}'.format(endofline, i))

        if bd.blockDataUsage == 0:
            infid.write('{1}{0}'.format(endofline, bd.blockDataUsage))
            continue

        elif bd.blockDataUsage == 1:
            fname = '{}{}{}.dat'.format(fileprefix, suffix_blockData, i)
            blockdata.writeBlockData(bd, dirname + '/' + fname)
            if verbose == 2:
                infid.write('1 {1} // block data file{0}'.format(endofline, fname))
            else:
                infid.write('1 {1}{0}'.format(endofline, fname))

    infid.write('{0}'.format(endofline))

    # Maximal scan fraction
    if verbose > 0:
        infid.write('\
/* MAXIMAL SCAN FRACTION FOR EACH TI (as many number(s) as number of training image(s)) */{0}'.format(endofline))

    for v in deesse_input.maxScanFraction:
        infid.write('{1}{0}'.format(endofline, v))

    infid.write('{0}'.format(endofline))

    # Pyramids
    if verbose > 0:
        infid.write('\
/* PYRAMIDS */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* I. PYRAMID GENERAL PARAMETERS:{0}\
      I.1. Number of pyramid level(s) (in addition to original simulation grid, i.e. number of{0}\
           reduction operations), integer (npyramidLevel):{0}\
              - = 0: no use of pyramids{0}\
              - > 0: use pyramids, npyramidLevel should be equal to the max of "nlevel" entries{0}\
                     in pyramid parameters for every variable (point II.1 below);{0}\
                     pyramid levels are indexed from fine to coarse:{0}\
                        * index 0            : original simulation grid{0}\
                        * index npyramidLevel: coarsest level{0}\
      If npyramidLevel > 0:{0}\
         I.2. for each level, i.e. for i = 1,..., npyramidLevel:{0}\
                 - kx, ky, kz (3 integer): reduction step along x,y,z-direction for the i-th reduction:{0}\
                      k[x|y|z] = 0: nothing is done, same dimension in output{0}\
                      k[x|y|z] = 1: same dimension in output (with weighted average over 3 nodes){0}\
                      k[x|y|z] = 2: classical gaussian pyramid{0}\
                      k[x|y|z] > 2: generalized gaussian pyramid{0}\
         I.3. pyramid simulation mode, key work (pyramidSimulationMode):{0}\
                 - PYRAMID_SIM_HIERARCHICAL:{0}\
                      (a) spreading conditioning data through the pyramid by simulation at each{0}\
                          level, from fine to coarse, conditioned to the level one rank finer{0}\
                      (b) simulation at the coarsest level, then simulation of each level, from coarse{0}\
                          to fine, conditioned to the level one rank coarser{0}\
                 - PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION:{0}\
                      (a) spreading conditioning data through the pyramid by simulation at each{0}\
                          level, from fine to coarse, conditioned to the level one rank finer{0}\
                      (b) simulation at the coarsest level, then simulation of each level, from coarse{0}\
                          to fine, conditioned to the gaussian expansion of the level one rank coarser{0}\
                 - PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE:{0}\
                      co-simulation of all levels, simulation done at one level at a time{0}\
         I.4. Factors to adapt the maximal number of neighboring nodes:{0}\
              I.4.1. Setting mode, key word (factorNneighboringNodeSettingMode):{0}\
                        - PYRAMID_NNEIGHBOR_ADAPTING_FACTOR_DEFAULT: set by default{0}\
                        - PYRAMID_NNEIGHBOR_ADAPTING_FACTOR_MANUAL : read in input{0}\
              If factorNneighboringNodeSettingMode == PYRAMID_NNEIGHBOR_ADAPTING_FACTOR_MANUAL:{0}\
                 I.4.2. The factors, depending on pyramid simulation mode:{0}\
                    - if pyramidSimulationMode == PYRAMID_SIM_HIERARCHICAL{0}\
                      or PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION:{0}\
                         - faCond[0], faSim[0], fbCond[0], fbSim[0],{0}\
                           ...,{0}\
                           faCond[n-1], faSim[n-1], fbCond[n-1], fbSim[n-1],{0}\
                           fbSim[n]:{0}\
                              I.e. (4*n+1) positive numbers where n = npyramidLevel, with the following{0}\
                              meaning. The maximal number of neighboring nodes (according to each variable){0}\
                              is multiplied by{0}\
                                (a) faCond[j] and faSim[j] for the conditioning level (level j){0}\
                                    and the simulated level (level j+1) resp. during step (a) above{0}\
                                (b) fbCond[j] and fbSim[j] for the conditioning level (level j+1) (expanded{0}\
                                    if pyramidSimulationMode == PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION){0}\
                                    and the simulated level (level j) resp. during step (b) above{0}\
                    - if pyramidSimulationMode == PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE:{0}\
                         - f[0],..., f[npyramidLevel-1], f[npyramidLevel]:{0}\
                              I.e. (npyramidLevel + 1) positive numbers, with the following meaning. The{0}\
                              maximal number of neighboring nodes (according to each variable) is{0}\
                              multiplied by f[j] for the j-th pyramid level.{0}\
         I.5. Factors to adapt the distance threshold (similar to I.4):{0}\
              I.5.1. Setting mode, key word (factorDistanceThresholdSettingMode):{0}\
                        - PYRAMID_THRESHOLD_ADAPTING_FACTOR_DEFAULT: set by default{0}\
                        - PYRAMID_THRESHOLD_ADAPTING_FACTOR_MANUAL : read in input{0}\
              If factorDistanceThresholdSettingMode == PYRAMID_THRESHOLD_ADAPTING_FACTOR_MANUAL:{0}\
                 I.5.2. The factors, depending on pyramid simulation mode (similar to I.4.2).{0}\
         I.6. Factors to adapt the maximal scan fraction:{0}\
              I.6.1. Setting mode, key word (factorMaxScanFractionSettingMode):{0}\
                        - PYRAMID_MAX_SCAN_FRACTION_ADAPTING_FACTOR_DEFAULT: set by default{0}\
                        - PYRAMID_MAX_SCAN_FRACTION_ADAPTING_FACTOR_MANUAL : read in input{0}\
              If factorMaxScanFractionSettingMode == PYRAMID_MAX_SCAN_FRACTION_ADAPTING_FACTOR_MANUAL:{0}\
                 I.6.2. The factors:{0}\
                    - f[0],..., f[npyramidLevel-1], f[npyramidLevel]:{0}\
                         I.e. (npyramidLevel + 1) positive numbers, with the following meaning. The{0}\
                      maximal scan fraction (according to each training image) is{0}\
                      multiplied by f[j] for the j-th pyramid level.{0}\
{0}\
         II. PYRAMID PARAMETERS FOR EACH VARIABLE:{0}\
{0}\
         II.1. nlevel: number of pyramid level(s) (number of reduction operations){0}\
                          - = 0: no use of pyramid for the considered variable{0}\
                          - > 0: use pyramids for the considered variable, with nlevel level{0}\
{0}\
         If nlevel > 0:{0}\
            II.2. Pyramid type, key word (pyramidType):{0}\
                     - PYRAMID_CONTINUOUS        : pyramid applied to continuous variable (direct){0}\
                     - PYRAMID_CATEGORICAL_AUTO  : pyramid for categorical variable{0}\
                                                      - pyramid for indicator variable of each category{0}\
                                                        except one (one pyramid per indicator variable){0}\
                     - PYRAMID_CATEGORICAL_CUSTOM: pyramid for categorical variable{0}\
                                                      - pyramid for indicator variable of each class{0}\
                                                        of values given explicitly (one pyramid per{0}\
                                                        indicator variable){0}\
                     - PYRAMID_CATEGORICAL_TO_CONTINUOUS:{0}\
                                                   pyramid for categorical variable{0}\
                                                      - the variable is transformed to a continuous{0}\
                                                        variable (according to connection between adjacent{0}\
                                                        nodes, the new values are ordered such that close{0}\
                                                        values correspond to the most connected categories),{0}\
                                                        then one pyramid for the transformed variable{0}\
                                                        is used{0}\
            If pyramidType == PYRAMID_CATEGORICAL_CUSTOM:{0}\
               II.3.  The classes of values (for which the indicator variables are{0}\
                  considered for pyramids) have to be defined; a class of values is given by a union{0}\
                  of interval(s): [inf_1,sup_1[ U ... U [inf_n,sup_n[.{0}\
                  Here are given:{0}\
                     - nclass: number of classes of values{0}\
                     - for i in 1,..., nclass: definition of the i-th class of values:{0}\
                          - ninterval: number of interval(s){0}\
                          - inf_1 sup_1 ... inf_ninterval sup_ninterval: inf and sup for each interval{0}\
                               these values should satisfy inf_i < sup_i{0}\
                  Then, for each class, the number of pyramid levels (number of reduction operations) is{0}\
                     - nlevel_i (for i in 1,..., nclass){0}\
*/{0}'.format(endofline))

    if verbose > 0:
        infid.write('\
/* PYRAMID GENERAL PARAMETERS */{0}'.format(endofline))

    gp = deesse_input.pyramidGeneralParameters

    if gp.npyramidLevel:
        if verbose == 2:
            infid.write('{1} // number of level(s) additional to initial SG{0}'.format(endofline, gp.npyramidLevel))
        else:
            infid.write('{1}{0}'.format(endofline, gp.npyramidLevel))

        for j, (jx, jy, jz) in enumerate(zip(gp.kx, gp.ky, gp.kz)):
            if verbose == 2:
                infid.write('{1} {2} {3} // reduction step along x, y, z for level {4}{0}'.format(endofline, jx, jy, jz, j))
            else:
                infid.write('{1} {2} {3}{0}'.format(endofline, jx, jy, jz))

        if gp.pyramidSimulationMode == 'hierarchical':
            if verbose == 2:
                infid.write('PYRAMID_SIM_HIERARCHICAL // pyramid simulation mode{0}'.format(endofline))
            else:
                infid.write('PYRAMID_SIM_HIERARCHICAL{0}'.format(endofline))
        elif gp.pyramidSimulationMode == 'hierarchical_using_expansion':
            if verbose == 2:
                infid.write('PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION // pyramid simulation mode{0}'.format(endofline))
            else:
                infid.write('PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION{0}'.format(endofline))
        elif gp.pyramidSimulationMode == 'all_level_one_by_one':
            if verbose == 2:
                infid.write('PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE // pyramid simulation mode{0}'.format(endofline))
            else:
                infid.write('PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE{0}'.format(endofline))
        else:
            if verbose == 2:
                infid.write('PYRAMID_SIM_NONE // pyramid simulation mode{0}'.format(endofline))
            else:
                infid.write('PYRAMID_SIM_NONE{0}'.format(endofline))

        if verbose == 2:
            infid.write('PYRAMID_NNEIGHBOR_ADAPTING_FACTOR_MANUAL // mode for adapting factors (max number of neighbors){0}'.format(endofline))
        else:
            infid.write('PYRAMID_NNEIGHBOR_ADAPTING_FACTOR_MANUAL{0}'.format(endofline))

        if gp.pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            for i in range(gp.npyramidLevel):
                if verbose == 2:
                    infid.write('{1} {2} {3} {4} // faCond[{5}], faSim[{5}], fbCond[{5}], fbSim[{5}]{0}'.format(endofline,
                        gp.factorNneighboringNode[4*i], gp.factorNneighboringNode[4*i+1],
                        gp.factorNneighboringNode[4*i+2], gp.factorNneighboringNode[4*i+3], i))
                else:
                    infid.write('{1} {2} {3} {4}{0}'.format(endofline,
                        gp.factorNneighboringNode[4*i], gp.factorNneighboringNode[4*i+1],
                        gp.factorNneighboringNode[4*i+2], gp.factorNneighboringNode[4*i+3]))

            if verbose == 2:
                infid.write('{1} // fbSim[{2}]{0}'.format(endofline, gp.factorNneighboringNode[4*gp.npyramidLevel], gp.npyramidLevel))
            else:
                infid.write('{1}{0}'.format(endofline, gp.factorNneighboringNode[4*gp.npyramidLevel]))

        elif gp.pyramidSimulationMode == 'all_level_one_by_one':
            for i, f in enumerate(gp.factorNneighboringNode):
                if verbose == 2:
                    infid.write('{1} // f[{2}]{0}'.format(endofline, f, i))
                else:
                    infid.write('{1}{0}'.format(endofline, f))

        if verbose == 2:
            infid.write('PYRAMID_THRESHOLD_ADAPTING_FACTOR_MANUAL // mode for adapting factors (distance threshold){0}'.format(endofline))
        else:
            infid.write('PYRAMID_THRESHOLD_ADAPTING_FACTOR_MANUAL{0}'.format(endofline))

        if gp.pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            for i in range(gp.npyramidLevel):
                if verbose == 2:
                    infid.write('{1} {2} {3} {4} // faCond[{5}], faSim[{5}], fbCond[{5}], fbSim[{5}]{0}'.format(endofline,
                        gp.factorDistanceThreshold[4*i], gp.factorDistanceThreshold[4*i+1],
                        gp.factorDistanceThreshold[4*i+2], gp.factorDistanceThreshold[4*i+3], i))
                else:
                    infid.write('{1} {2} {3} {4}{0}'.format(endofline,
                        gp.factorDistanceThreshold[4*i], gp.factorDistanceThreshold[4*i+1],
                        gp.factorDistanceThreshold[4*i+2], gp.factorDistanceThreshold[4*i+3]))

            if verbose == 2:
                infid.write('{1} // fbSim[{2}]{0}'.format(endofline, gp.factorDistanceThreshold[4*gp.npyramidLevel], gp.npyramidLevel))
            else:
                infid.write('{1}{0}'.format(endofline, gp.factorDistanceThreshold[4*gp.npyramidLevel]))

        elif gp.pyramidSimulationMode == 'all_level_one_by_one':
            for i, f in enumerate(gp.factorDistanceThreshold):
                if verbose == 2:
                    infid.write('{1} // f[{2}]{0}'.format(endofline, f, i))
                else:
                    infid.write('{1}{0}'.format(endofline, f))

        if verbose == 2:
            infid.write('PYRAMID_MAX_SCAN_FRACTION_ADAPTING_FACTOR_MANUAL // mode for adapting factors (maximal scan fraction){0}'.format(endofline))
        else:
            infid.write('PYRAMID_MAX_SCAN_FRACTION_ADAPTING_FACTOR_MANUAL{0}'.format(endofline))

        for i, f in enumerate(gp.factorMaxScanFraction):
            if verbose == 2:
                infid.write('{1} // f[{2}]{0}'.format(endofline, f, i))
            else:
                infid.write('{1}{0}'.format(endofline, f))

        for i, pp in enumerate(deesse_input.pyramidParameters):
            if verbose > 0:
                infid.write('\
/* PYRAMID PARAMETERS FOR VARIABLE #{1} */{0}'.format(endofline, i))

            if verbose == 2:
                infid.write('{1} // nlevel{0}'.format(endofline, pp.nlevel))
            else:
                infid.write('{1}{0}'.format(endofline, pp.nlevel))

            if pp.pyramidType == 'continuous':
                if verbose == 2:
                    infid.write('PYRAMID_CONTINUOUS // pyramid type{0}'.format(endofline))
                else:
                    infid.write('PYRAMID_CONTINUOUS{0}'.format(endofline))
            elif pp.pyramidType == 'categorical_auto':
                if verbose == 2:
                    infid.write('PYRAMID_CATEGORICAL_AUTO // pyramid type{0}'.format(endofline))
                else:
                    infid.write('PYRAMID_CATEGORICAL_AUTO{0}'.format(endofline))
            elif pp.pyramidType == 'categorical_custom':
                if verbose == 2:
                    infid.write('PYRAMID_CATEGORICAL_CUSTOM // pyramid type{0}'.format(endofline))
                else:
                    infid.write('PYRAMID_CATEGORICAL_CUSTOM{0}'.format(endofline))

                if verbose == 2:
                    infid.write('{1} // nclass{0}'.format(endofline, pp.nclass))
                else:
                    infid.write('{1}{0}'.format(endofline, pp.nclass))

                for j, ci in enumerate(pp.classInterval):
                    infid.write('{}  '.format(len(ci)))
                    for inter in ci:
                        infid.write(' {} {}'.format(inter[0], inter[1]))

                    if verbose == 2:
                        infid.write(' // class #{1} (ninterval, and interval(s)){0}'.format(endofline, j))
                    else:
                        infid.write('{0}'.format(endofline))
            elif pp.pyramidType == 'categorical_to_continuous':
                if verbose == 2:
                    infid.write('PYRAMID_CATEGORICAL_TO_CONTINUOUS // pyramid type{0}'.format(endofline))
                else:
                    infid.write('PYRAMID_CATEGORICAL_TO_CONTINUOUS{0}'.format(endofline))
            else:
                if verbose == 2:
                    infid.write('PYRAMID_NONE // pyramid type{0}'.format(endofline))
                else:
                    infid.write('PYRAMID_NONE{0}'.format(endofline))

    else: # deesse_input.pyramidGeneralParameters.npyramidLevel (gp.npyramidLevel) == 0
        infid.write('{1}{0}'.format(endofline, gp.npyramidLevel))

    infid.write('{0}'.format(endofline))

    # Tolerance
    if verbose > 0:
        infid.write('\
/* TOLERANCE */{0}'.format(endofline))

    if verbose == 2:
        infid.write("\
/* Tolerance t on the threshold value for flagging nodes (for post-processing):{0}\
   let d(i) be the distance between the data event in the simulation grid and in the training{0}\
   image for the i-th variable and s(i) be the distance threshold for the i-th variable, and let{0}\
   e(i) = max(0, (d(i)-s(i))/s(i)) be the relative error for the i-th variable, i.e. the relative{0}\
   part of the distance d(i) beyond the threshold s(i); during the scan of the training image, a node{0}\
   that minimizes e = sum (e(i)) is retained (the scan is stopped if e = 0); if e is greater than the{0}\
   tolerance t (given here), then the current simulated node and all non-conditioning nodes of the{0}\
   data events (one per variable) in the simulation grid are flagged for resimulation (post-processing).{0}\
   Note that if probability / connectivity / block data constraints used a similar error as e(i) that{0}\
   contributes in the sum defining the error e.{0}\
*/{0}".format(endofline))

    infid.write('{1}{0}{0}'.format(endofline, deesse_input.tolerance))

    # Post-processing
    if verbose > 0:
        infid.write('\
/* POST-PROCESSING */{0}'.format(endofline))

    if verbose == 2:
        infid.write('\
/* 1. Maximal number of path(s) (npostProcessingPathMax){0}\
   2. If npostProcessingPathMax > 0:{0}\
      key word for post-processing parameters (i. e. number of neighboring nodes, distance threshold,{0}\
      maximal scan fraction, and tolerance):{0}\
         - POST_PROCESSING_PARAMETERS_DEFAULT: for default parameters{0}\
         - POST_PROCESSING_PARAMETERS_SAME   : for same parameters as given above{0}\
         - POST_PROCESSING_PARAMETERS_MANUAL : for manual settings{0}\
   3. If npostProcessingPathMax > 0 and POST_PROCESSING_PARAMETERS_MANUAL:{0}\
         MAXIMAL NUMBER OF NEIGHBORING NODES FOR EACH VARIABLE (as many number(s) as number of variable(s)){0}\
         MAXIMAL DENSITY OF NEIGHBORING NODES IN SEARCH NEIGHBORHOOD FOR EACH VARIABLE (as many number(s){0}\
            as number of variable(s)){0}\
         DISTANCE THRESHOLD FOR EACH VARIABLE (as many number(s) as number of variable(s)){0}\
         MAXIMAL SCAN FRACTION FOR EACH TI (as many number(s) as number of training image(s)){0}\
         TOLERANCE{0}\
*/{0}'.format(endofline))

    infid.write('{1}{0}'.format(endofline, deesse_input.npostProcessingPathMax))

    if deesse_input.npostProcessingPathMax:
        infid.write('{1}{0}'.format(endofline, 'POST_PROCESSING_PARAMETERS_MANUAL'))

        if verbose > 0:
            infid.write('\
/* POST-PROCESSING: MAXIMAL NUMBER OF NEIGHBORING NODES FOR EACH VARIABLE */{0}'.format(endofline))

        for v in deesse_input.postProcessingNneighboringNode:
            infid.write('{1}{0}'.format(endofline, v))
        infid.write('{0}'.format(endofline))

        if verbose > 0:
            infid.write('\
/* POST-PROCESSING: MAXIMAL DENSITY OF NEIGHBORING NODES IN SEARCH NEIGHBORHOOD FOR EACH VARIABLE */{0}'.format(endofline))

        for v in deesse_input.postProcessingNeighboringNodeDensity:
            infid.write('{1}{0}'.format(endofline, v))
        infid.write('{0}'.format(endofline))

        if verbose > 0:
            infid.write('\
/* POST-PROCESSING: DISTANCE THRESHOLD FOR EACH VARIABLE */{0}'.format(endofline))

        for v in deesse_input.postProcessingDistanceThreshold:
            infid.write('{1}{0}'.format(endofline, v))
        infid.write('{0}'.format(endofline))

        if verbose > 0:
            infid.write('\
/* POST-PROCESSING: MAXIMAL SCAN FRACTION FOR EACH TI */{0}'.format(endofline))

        for v in deesse_input.postProcessingMaxScanFraction:
            infid.write('{1}{0}'.format(endofline, v))
        infid.write('{0}'.format(endofline))

        if verbose > 0:
            infid.write('\
/* POST-PROCESSING: TOLERANCE */{0}'.format(endofline))

        infid.write('{1}{0}{0}'.format(endofline, deesse_input.postProcessingTolerance))

    else: # no post processing
        infid.write('{0}'.format(endofline))

    # Seed number and seed increment
    if verbose > 0:
        infid.write('\
/* SEED NUMBER AND SEED INCREMENT */{0}'.format(endofline))

    infid.write('{1}{0}{2}{0}{0}'.format(endofline, deesse_input.seed, deesse_input.seedIncrement))

    # Number of realization(s)
    if verbose > 0:
        infid.write('\
/* NUMBER OF REALIZATION(S) */{0}'.format(endofline))

    infid.write('{1}{0}{0}'.format(endofline, deesse_input.nrealization))

    # END
    infid.write('END{0}'.format(endofline))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class DeesseEstimator():
    """ DS estimator: scikit-learn compatible wrapper for DeesseInput
    and DeesseRun
    """

    def __init__(self, varnames=None, nthreads=-1, **kwargs):
        """
        :param varnames: must be specified, list of all variables
            (including X, Y, Z) in the conditioning data
        :param kwargs: parameters of DeesseInput
        """
        if varnames is None:
            raise ValueError("Please specify varnames: list of all variables in the observation set")
        self.deesse_parameters = kwargs
        self.varnames = varnames
        self.nthreads = nthreads

    def set_params(self, **parameters):
        """
        Sets simulation parameters according to a dictionary
        for compatibility with scikit-learn
        """
        for parameter, value in parameters.items():
            self.deesse_parameters[parameter] = value
        return self

    def get_params(self, deep=True):
        """
        Returns all parameters in a dictionary fo compatibility with scikit-learn
        """
        return {'varnames': self.varnames, **self.deesse_parameters, 'nthreads': self.nthreads}

    def fit(self, X, y):
        """An implementation of DS fitting function.
        Set ups all parametes and reads the TI for the DS.
        Constructs DS input. Converts X,y into hard data

        :param X : array-like (provides __array__ method) shape (n_samples, n_features)
            The training input samples.
        :param y : array-like, shape (n_samples,)
            The target values (class labels)

        :return self: returns self
        """

        # Convert X,y into the hard conditioning Point
        if y.ndim == 1:
            y = y[:, np.newaxis]
        array = np.hstack([X, y])
        self.hd = PointSet(npt=array.shape[0],
                nv=array.shape[1],
                val=array.transpose(),
                varname=self.varnames)

        self.deesse_input = DeesseInput(dataPointSet=self.hd, **self.deesse_parameters)

        # set properties for compatibility with scikit-learn
        self.classes_,y = np.unique(y, return_inverse=True)
        self.is_fitted_ = True
        self.X_ = X
        self.y_ = y

        # predict_proba remembers last prediction to avoid recomputing it
        # for multiple custom scorers
        # fitting invalidates last prediction
        try:
            del self.previous_X_
            del self.previous_y_
        except AttributeError:
            pass

        # `fit` should always return `self`
        return self

    def simulate(self, verbose=0, unconditional=False):
        """
        Return DeeSse simulation

        :param verbose=0: (int) 0, 1, 2 specifies verbosity of deesseRun
        :param unconditional=False: if True, performs unconditional simulation
            ignores the fitted parameters

        :return deesse_output:  (dict) {'sim':sim, 'path':path, 'error':error}
            With nreal = deesse_input.nrealization:
            sim:    (1-dimensional array of Img (class) of size nreal)
                        sim[i]: i-th realisation
            path:   (1-dimensional array of Img (class) of size nreal or None)
                        path[i]: path index map for the i-th realisation
                        (path is None if deesse_input.outputPathIndexFlag is False)
            error:  (1-dimensional array of Img (class) of size nreal or None)
                        error[i]: error map for the i-th realisation
                        (path is None if deesse_input.outputErrorFlag is False)
        """
        if unconditional is True:
            deesse_input = DeesseInput(**self.deesse_parameters)
        else:
            try:
                deesse_input = self.deesse_input
            except AttributeError:
                deesse_input = DeesseInput(**self.deesse_parameters)

        return deesseRun(deesse_input, verbose=verbose, nthreads=self.nthreads)


# ----------------------------------------------------------------------------
class DeesseRegressor(DeesseEstimator):
    def predict(self, X):
        X = X.__array__()

        deesse_output = self.simulate()

    def sample_y(self, X):
        """ Implementation of a predicting function, probabilities for each category.
        Uses pixel-wise average proportion of DS predictions.
        Number od DS simulations corresponds to number of realisations.

        :param X: array-like (must implement __array__ method)
            containing spatial coordinates

        :return y:  (ndarray) probability predictions
            shape (n_samples, n_features)
        """
        X = X.__array__()

        deesse_output = self.simulate()

        # compute pixel-wise proportions
        all_sim = img.gatherImages(deesse_output['sim'])

        p = self.deesse_parameters
        # get only relevant pixels for comparison
        y = np.zeros((X.shape[0], p['nrealization']))
        for counter, point in enumerate(X):
            # get index of predicted point
            index = img.pointToGridIndex(point[0], point[1], point[2],
                    nx=p['nx'], ny=p['ny'], nz=p['nz'],
                    sx=p['sx'], sy=p['sy'], sz=p['sz'],
                    ox=p['ox'], oy=p['oy'], oz=p['oz'])
            y[counter, :] = all_sim.val[:,
                                              index[2],
                                              index[1],
                                              index[0]]

        # This is a workaround for scorers to reuse results
        # because our scorers call predict_proba always
        # we want to reuse the results for the same X if no fitting
        # was done in the meantime
        self.previous_X_ = X
        self.previous_y_ = y

        return y
# ----------------------------------------------------------------------------
class DeesseClassifier(DeesseEstimator):
    def predict(self, X):
        """ Implementation of a predicting function.
        Returns predicted facies by taking the biggest probability
        eveluated by the predict_proba method

        :param X: array-like (must implement __array__ method)
            containing spatial coordinates

        :return y:  (ndarray) predicted classes
            shape (n_samples, )
        """
        y = self.predict_proba(X)
        return self.classes_.take(np.argmax(y, axis=1), axis=0)

    def predict_proba(self, X):
        """ Implementation of a predicting function, probabilities for each category.
        Uses pixel-wise average proportion of DS predictions.
        Number od DS simulations corresponds to number of realisations.

        :param X: array-like (must implement __array__ method)
            containing spatial coordinates

        :return y:  (ndarray) probability predictions
            shape (n_samples, n_features)
        """
        X = X.__array__()

        # suppress print bevause license check is still printed :(
        deesse_output = self.simulate()

        # compute pixel-wise proportions
        all_sim = img.gatherImages(deesse_output['sim'])
        all_sim_stats = img.imageCategProp(all_sim, self.classes_)

        p = self.deesse_parameters
        # get only relevant pixels for comparison
        y = np.zeros((X.shape[0], len(self.classes_)))
        for counter, point in enumerate(X):
            # get index of predicted point
            index = img.pointToGridIndex(point[0], point[1], point[2],
                    nx=p['nx'], ny=p['ny'], nz=p['nz'],
                    sx=p['sx'], sy=p['sy'], sz=p['sz'],
                    ox=p['ox'], oy=p['oy'], oz=p['oz'])
            # retrieve the facies probability
            for i in range(len(self.classes_)):
                y[counter, i] = all_sim_stats.val[i,
                                                  index[2],
                                                  index[1],
                                                  index[0]]

        # This is a workaround for scorers to reuse results
        # because our scorers call predict_proba always
        # we want to reuse the results for the same X if no fitting
        # was done in the meantime
        self.previous_X_ = X
        self.previous_y_ = y

        return y
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.deesseinterface'.")
