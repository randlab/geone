#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'deesseinterface.py'
author:         Julien Straubhaar
date:           jan-2018

Module interfacing deesse for python.
"""

import numpy as np
import sys, os, re, copy
import multiprocessing

from geone import img, blockdata
from geone.deesse_core import deesse
from geone.img import Img, PointSet
from geone.blockdata import BlockData

version = [deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER]

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
                            search radii rx, ry, rz, in number of cells, are
                            explicitly given

        rx, ry, rz: (floats) radii, in number of cells, in each direction
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

    # ------------------------------------------------------------------------
    def __repr__(self):
        out = '*** SearchNeighborhoodParameters object ***'
        out = out + '\n' + 'radiusMode = {0.radiusMode}'.format(self)
        if self.radiusMode == 'manual':
            out = out + '\n' + '(rx, ry, rz) = ({0.rx}, {0.ry}, {0.rz})'.format(self)
        out = out + '\n' + 'anisotropyRatioMode = {0.anisotropyRatioMode}'.format(self)
        if self.anisotropyRatioMode == 'manual':
            out = out + '\n' + '(ax, ay, az) = ({0.ax}, {0.ay}, {0.az})'.format(self)
        out = out + '\n' + '(angle1, angle2, angle3) = ({0.angle1}, {0.angle2}, {0.angle3}) # (azimuth, dip, plunge)'.format(self)
        out = out + '\n' + 'power = {0.power}'.format(self)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class SoftProbability(object):
    """
    Defines probability constraints (for one variable):
        probabilityConstraintUsage:
                    (int) indicates the usage of probability constraints:
                        - 0: no probability constraint
                        - 1: global probability constraints
                        - 2: local probability constraints using support
                        - 3: local probability constraints based on rejection

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
                        of the SG, used when probabilityConstraintUsage in [2, 3]

        localPdfSupportRadius:
                    (float) support radius for local pdf, used when
                        probabilityConstraintUsage == 2

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
                        used when robabilityConstraintUsage in [1, 2]

        rejectionMode:
                    (int) indicates the mode of rejection (during the scan of
                        the training image):
                            - 0: rejection is done first (before checking pattern
                                (and other constraint)) according to acceptation
                                probabilities proportional to p[i]/q[i] (for
                                class i), where
                                    - q is the marginal pdf of the scanned
                                        training image
                                    - p is the given local pdf at the simulated
                                        node
                            - 1: rejection is done last (after checking pattern
                                (and other constraint)) according to acceptation
                                probabilities proportional to p[i] (for class i),
                                where
                                    - p is the given local pdf at the simulated
                                        node
                                method used when probabilityConstraintUsage == 3

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
                        used when robabilityConstraintUsage in [1, 2]

        constantThreshold:
                    (float) (acceptance) threshold value for pdf's comparison,
                        used when robabilityConstraintUsage in [1, 2] and
                        probabilityConstraintThresholdType == 0

        dynamicThresholdParameters:
                    (1-dimensional array of floats of size 7) parameters for
                        dynamic threshold (used for pdf's comparison),
                        used when probabilityConstraintUsage in [1, 2] and
                        probabilityConstraintThresholdType == 1
    """

    def __init__(self,
                 probabilityConstraintUsage=0,
                 nclass=0,
                 classInterval=None,
                 globalPdf=None,
                 localPdf=None,
                 localPdfSupportRadius=12.0,
                 localCurrentPdfComputation=0,
                 comparingPdfMethod=5,
                 rejectionMode=0,
                 deactivationDistance=4.0,
                 probabilityConstraintThresholdType=0,
                 constantThreshold=1.e-3,
                 dynamicThresholdParameters=None):

        fname = 'SoftProbability'

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
                print(f'ERROR ({fname}): field "globalPdf"...')
                return None

        if localPdf is None:
            self.localPdf = None
        else:
            self.localPdf = np.asarray(localPdf, dtype=float)

        self.localPdfSupportRadius = localPdfSupportRadius
        self.localCurrentPdfComputation = localCurrentPdfComputation
        self.comparingPdfMethod = comparingPdfMethod
        self.rejectionMode = rejectionMode
        self.deactivationDistance = deactivationDistance
        self.probabilityConstraintThresholdType = probabilityConstraintThresholdType
        self.constantThreshold = constantThreshold
        self.dynamicThresholdParameters = dynamicThresholdParameters

    # ------------------------------------------------------------------------
    def __repr__(self):
        out = '*** SoftProbability object ***'
        out = out + '\n' + 'probabilityConstraintUsage = {0.probabilityConstraintUsage}:'.format(self)
        if self.probabilityConstraintUsage == 0:
            out = out + ' no probability constraint'
        elif self.probabilityConstraintUsage == 1:
            out = out + ' global probability constraint'
        elif self.probabilityConstraintUsage == 2:
            out = out + ' local probability constraint using support'
        elif self.probabilityConstraintUsage == 3:
            out = out + ' local probability constraint based on rejection'
        else:
            out = out + ' unknown'
        if self.probabilityConstraintUsage in [1, 2, 3]:
            out = out + '\n' + 'nclass = {0.nclass} # number of classes'.format(self)
            for j, ci in enumerate(self.classInterval):
                out = out + '\n' + 'class {}: '.format(j)
                for k, inter in enumerate(ci):
                    if k > 0:
                        out = out + ' U '
                    out = out + str(inter)
        if self.probabilityConstraintUsage == 1:
            out = out + '\n' + 'global pdf: ' + str(self.globalPdf)
        elif self.probabilityConstraintUsage in [2, 3]:
            out = out + '\n' + 'local pdf: array ".localPdf"'
        if self.probabilityConstraintUsage == 2:
            out = out + '\n' + 'localPdfSupportRadius = {0.localPdfSupportRadius} # support radius'.format(self)
            out = out + '\n' + 'localCurrentPdfComputation = {0.localCurrentPdfComputation}:'.format(self)
            if self.localCurrentPdfComputation == 0:
                out = out + ' complete'
            elif self.localCurrentPdfComputation == 1:
                out = out + ' approximate'
            else:
                out = out + ' unknown'
        if self.probabilityConstraintUsage in [1, 2]:
            out = out + '\n' + 'comparingPdfMethod = {0.comparingPdfMethod}:'.format(self)
            if self.comparingPdfMethod == 0:
                out = out + ' MAE (Mean Absolute Error)'
            elif self.comparingPdfMethod == 1:
                out = out + ' RMSE (Root Mean Squared Error)'
            elif self.comparingPdfMethod == 2:
                out = out + ' KLD (Kullback Leibler Divergence)'
            elif self.comparingPdfMethod == 3:
                out = out + ' JSD (Jensen-Shannon Divergence)'
            elif self.comparingPdfMethod == 4:
                out = out + ' MLikRsym (Mean Likelihood Ratio (over over class, symmetric target interval))'
            elif self.comparingPdfMethod == 5:
                out = out + ' MLikRopt (Mean Likelihood Ratio (over each class, optimal target interval))'
            else:
                out = out + ' unknown'
        if self.probabilityConstraintUsage == 3:
            out = out + '\n' + 'rejectionMode = {0.rejectionMode}:'.format(self)
            if self.rejectionMode == 0:
                out = out + ' rejection applied first'
            elif self.rejectionMode == 1:
                out = out + ' rejection applied last'
            else:
                out = out + ' unknown'
        if self.probabilityConstraintUsage in [1, 2, 3]:
            out = out + '\n' + 'deactivationDistance = {0.deactivationDistance} # deactivation distance'.format(self)
        if self.probabilityConstraintUsage in [1, 2]:
            out = out + '\n' + 'probabilityConstraintThresholdType = {0.probabilityConstraintThresholdType}:'.format(self)
            if self.probabilityConstraintThresholdType == 0:
                out = out + ' constant'
            elif self.probabilityConstraintThresholdType == 1:
                out = out + ' dynamic'
            else:
                out = out + ' unknown'
            if self.probabilityConstraintThresholdType == 0:
                out = out + '\n' + 'constantThreshold = {0.constantThreshold} # threshold value'.format(self)
            elif self.probabilityConstraintThresholdType == 1:
                out = out + '\n' + 'dynamic threshold parameters: ' + str(self.dynamicThresholdParameters)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
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

        nclass:
                (int) number of classes of values
                    (unused if connectivityConstraintUsage == 0)
        classInterval:
                (list of nclass 2-dimensional array of floats with 2 columns)
                    definition of the classes of values by intervals,
                    classInterval[i] is a (n_i, 2) array a, defining the
                    i-th class as the union of intervals:
                        [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][

        varname:
                (string) variable name for connected component label (should be
                    in a conditioning data set)
                    Note: label negative or zero means no connectivity constraint

        tiAsRefFlag:
                (bool) indicates that the (first) training image is used as
                    reference for connectivity (True) or that the reference image
                    for connectivity is given by refConnectivityImage (False,
                    possible only if connectivityConstraintUsage == 1 or 2)

        refConnectivityImage:
                (Img class, or None) reference image for connectivity
                    (used only if tiAsRefFlag is False)

        refConnectivityVarIndex:
                (int) variable index in image refConnectivityImage for
                    the search of connected paths (used only if tiAsRefFlag is
                    False)

        deactivationDistance:
                (float) deactivation distance (the connectivity constraint is
                    deactivated if the distance between the current
                    simulated node and the last node in its neighbors (used for
                    the search in the TI) (distance computed according to the
                    corresponding search neighborhood parameters) is below the
                    given deactivation distance), used when
                    connectivityConstraintUsage == 3

        threshold:
                (float) threshold value for connectivity patterns comparison,
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
                 deactivationDistance=0.0,
                 threshold=0.01):

        fname = 'Connectivity'

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
            print(f'ERROR ({fname}): field "refConnectivityImage"...')
            return None

        self.refConnectivityImage = refConnectivityImage
        self.refConnectivityVarIndex = refConnectivityVarIndex
        self.deactivationDistance = deactivationDistance
        self.threshold = threshold

    # ------------------------------------------------------------------------
    def __repr__(self):
        out = '*** Connectivity object ***'
        out = out + '\n' + 'connectivityConstraintUsage = {0.connectivityConstraintUsage}:'.format(self)
        if self.connectivityConstraintUsage == 0:
            out = out + ' no connectivity constraint'
        elif self.connectivityConstraintUsage == 1:
            out = out + ' set connecting paths (random order) before simulation'
        elif self.connectivityConstraintUsage == 2:
            out = out + ' set connecting paths ("smart" order) before simulation'
        elif self.connectivityConstraintUsage == 3:
            out = out + ' check connectivity pattern during simulation'
        else:
            out = out + ' unknown'
        if self.connectivityConstraintUsage in [1, 2, 3]:
            out = out + '\n' + 'connectivityType = {0.connectivityType}:'.format(self)
            if self.connectivityConstraintUsage == 'connect_face':
                out = out + ' 6-neighbors connection (by face)'
            elif self.connectivityConstraintUsage == 'connect_face_edge':
                out = out + ' 18-neighbors connection (by face or edge)'
            elif self.connectivityConstraintUsage == 'connect_face_edge_corner':
                out = out + ' 26-neighbors connection (by face, edge or corner)'
            else:
                out = out + ' unknown'
            out = out + '\n' + 'nclass = {0.nclass} # number of classes'.format(self)
            for j, ci in enumerate(self.classInterval):
                out = out + '\n' + 'class {}: '.format(j)
                for k, inter in enumerate(ci):
                    if k > 0:
                        out = out + ' U '
                    out = out + str(inter)
            out = out + '\n' + 'tiAsRefFlag = {0.tiAsRefFlag} # TI is used as referentce image for connecivity ?'.format(self)
            if not self.tiAsRefFlag:
                out = out + '\n' + 'refConnectivityImage = {0.refConnectivityImage}'.format(self)
                out = out + '\n' + 'refConnectivityVarIndex = {0.refConnectivityVarIndex}'.format(self)
            out = out + '\n' + 'deactivationDistance = {0.deactivationDistance} # deactivation distance'.format(self)
            out = out + '\n' + 'threshold = {0.threshold}'.format(self)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
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

        kx:     (1-dimensional array of ints of size npyramidLevel)
                    reduction step along x-direction for each level:
                    - kx[.] = 0: nothing is done, same dimension after reduction
                    - kx[.] = 1: same dimension after reduction
                                 (with weighted average over 3 nodes)
                    - kx[.] = 2: classical gaussian pyramid
                    - kx[.] > 2: generalized gaussian pyramid
                    (unused if npyramidLevel == 0)

        ky:     (1-dimensional array of ints of size npyramidLevel)
                    reduction step along y-direction for each level:
                    - ky[.] = 0: nothing is done, same dimension after reduction
                    - ky[.] = 1: same dimension after reduction
                                 (with weighted average over 3 nodes)
                    - ky[.] = 2: classical gaussian pyramid
                    - ky[.] > 2: generalized gaussian pyramid
                    (unused if npyramidLevel == 0)

        kz:     (1-dimensional array of ints of size npyramidLevel)
                    reduction step along z-direction for each level:
                    - kz[.] = 0: nothing is done, same dimension after reduction
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
                            (a) faCond[j] and faSim[j] for the conditioning level
                                (level j) and the simulated level (level j+1) resp.
                                during step (a) above
                            (b) fbCond[j] and fbSim[j] for the conditioning level
                                (level j+1) (expanded if pyramidSimulationMode ==
                                'hierarchical_using_expansion') and the simulated
                                level (level j) resp. during step (b) above
                    - if pyramidSimulationMode == all_level_one_by_one':
                        array of size npyramidLevel + 1 with entries:
                           f[0],...,f[npyramidLevel-1],f[npyramidLevel]
                        i.e. (npyramidLevel + 1) positive numbers, with the
                        following meaning. The maximal number of neighboring
                        nodes (according to each variable) is multiplied by f[j]
                        for the j-th pyramid level

        factorDistanceThreshold:
                (1-dimensional array of floats) factors for adpating the distance
                    (acceptance) threshold (similar to factorNneighboringNode)

        factorMaxScanFraction:
                (1-dimensional array of floats of size npyramidLevel + 1)
                    factors for adpating the maximal scan fraction: the maximal
                    scan fraction (according to each training image) is multiplied
                    by factorMaxScanFraction[j] for the j-th pyramid level
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

        fname = 'PyramidGeneralParameters'

        self.npyramidLevel = npyramidLevel

        # pyramidSimulationMode
        if pyramidSimulationMode not in ('hierarchical', 'hierarchical_using_expansion', 'all_level_one_by_one'):
            print(f'ERROR ({fname}): unknown pyramidSimulationMode')
            return None

        self.pyramidSimulationMode = pyramidSimulationMode

        if npyramidLevel > 0:
            # kx, ky, kz
            if kx is None:
                self.kx = np.array([2 * int (nx>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kx = np.asarray(kx, dtype='int').reshape(npyramidLevel)
                except:
                    print(f'ERROR ({fname}): field "kx"...')
                    return None

            if ky is None:
                self.ky = np.array([2 * int (ny>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.ky = np.asarray(ky, dtype='int').reshape(npyramidLevel)
                except:
                    print(f'ERROR ({fname}): field "ky"...')
                    return None

            if kz is None:
                self.kz = np.array([2 * int (nz>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kz = np.asarray(kz, dtype='int').reshape(npyramidLevel)
                except:
                    print(f'ERROR ({fname}): field "kz"...')
                    return None

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
                        print(f'ERROR ({fname}): field "factorNneighboringNode"...')
                        return None

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        print(f'ERROR ({fname}): field "factorDistanceThreshold"...')
                        return None

            else: # pyramidSimulationMode == 'all_level_one_by_one'
                n = npyramidLevel + 1
                if factorNneighboringNode is None:
                    factorNneighboringNode = 1./n * np.ones(n)
                    self.factorNneighboringNode = factorNneighboringNode
                else:
                    try:
                        self.factorNneighboringNode = np.asarray(factorNneighboringNode, dtype=float).reshape(n)
                    except:
                        print(f'ERROR ({fname}): field "factorNneighboringNode"...')
                        return None

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        print(f'ERROR ({fname}): field "factorDistanceThreshold"...')
                        return None

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
                    print(f'ERROR ({fname}): field "factorMaxScanFraction"...')
                    return None

        else: # npyramidLevel <= 0
            self.kx = None
            self.ky = None
            self.kz = None
            self.pyramidSimulationMode = None
            self.factorNneighboringNode = None
            self.factorDistanceThreshold = None
            self.factorMaxScanFraction = None

    # ------------------------------------------------------------------------
    def __repr__(self):
        out = '*** PyramidGeneralParameters object ***'
        out = out + '\n' + 'npyramidLevel = {0.npyramidLevel} # number of pyramid level(s) (in addition to original simulation grid)'.format(self)
        if self.npyramidLevel > 0:
            out = out + '\n' + 'kx = ' + str(self.kx) + ' # reduction factor along x-axis for each level'
            out = out + '\n' + 'ky = ' + str(self.ky) + ' # reduction factor along y-axis for each level'
            out = out + '\n' + 'kz = ' + str(self.kz) + ' # reduction factor along z-axis for each level'
            out = out + '\n' + 'pyramidSimulationMode = {0.pyramidSimulationMode}'.format(self)
            out = out + '\n' + 'factorNneighboringNode = ' + str(self.factorNneighboringNode)
            out = out + '\n' + 'factorDistanceThreshold = ' + str(self.factorDistanceThreshold)
            out = out + '\n' + 'factorMaxScanFraction = ' + str(self.factorMaxScanFraction)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class PyramidParameters(object):
    """
    Defines the parameters for pyramid for one variable:
        nlevel: (int) number of pyramid level(s) (in addition to original
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

        nclass: (int) number of classes of values
                    (used when pyramidType == 'categorical_custom')
        classInterval:
                (list of nclass 2-dimensional array of floats with 2 columns)
                    definition of the classes of values by intervals,
                    classInterval[i] is a (n_i, 2) array a, defining the
                    i-th class as the union of intervals:
                        [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][
                    (used when pyramidType == 'categorical_custom')

        outputLevelFlag:
                (1-dimensional array of 'bool', of size nlevel)
                    flag indicating which level is saved in output:
                    - outputLevelFlag[j]:
                        - False: level of index (j+1) will not be saved in output
                        - True: level of index (j+1) will be saved in output
                            (only the pyramid for the original variables flagged
                            for output in the field 'outputVarFlag' of the parent
                            class 'DeesseInput' will be saved)
                    - the name of the output variables are set to
                            <vname>_ind<i>_lev<k>_real<n>
                        where
                        - <vname> is the name of the "original" variable,
                        - <i> is a pyramid index for that variable which starts
                            at 0 (more than one index can be required if the
                            pyramid type is set to 'categorical_auto' or
                            'categorical_custom'),
                        - <k> is the level index,
                        - <n> is the realization index (starting from 0)
                    - the values of the output variables are the normalized
                        values (as used during the simulation in every level)
    """

    def __init__(self,
                 nlevel=0,
                 pyramidType='none',
                 nclass=0,
                 classInterval=None,
                 outputLevelFlag=None):

        fname = 'PyramidParameters'

        self.nlevel = nlevel

        if pyramidType not in ('none', 'continuous', 'categorical_auto', 'categorical_custom', 'categorical_to_continuous'):
            print(f'ERROR ({fname}): unknown pyramidType')
            return None

        self.pyramidType = pyramidType

        self.nclass = nclass
        self.classInterval = classInterval

        if outputLevelFlag is None:
            self.outputLevelFlag = np.array([False for i in range(nlevel)], dtype='bool') # set dtype='bool' in case of nlevel=0
        else:
            try:
                self.outputLevelFlag = np.asarray(outputLevelFlag, dtype='bool').reshape(nlevel)
            except:
                print(f'ERROR ({fname}): field "outputLevelFlag"...')
                return None

    # ------------------------------------------------------------------------
    def __repr__(self):
        out = '*** PyramidParameters object ***'
        out = out + '\n' + 'nlevel = {0.nlevel} # number of pyramid level(s) (in addition to original simulation grid)'.format(self)
        if self.nlevel > 0:
            out = out + '\n' + 'pyramidType = {0.pyramidType}'.format(self)
            if self.pyramidType == 'categorical_custom':
                out = out + '\n' + 'nclass = {0.nclass} # number of classes'.format(self)
                for j, ci in enumerate(self.classInterval):
                    out = out + '\n' + 'class {}: '.format(j)
                    for k, inter in enumerate(ci):
                        if k > 0:
                            out = out + ' U '
                        out = out + str(inter)
            out = out + '\n' + 'outputLevelFlag = {0.outputLevelFlag}'.format(self)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class DeesseInput(object):
    """
    Defines deesse input:
        simName:    (str) simulation name (not useful)
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

        nTI:        (int) number of training image(s) (TI)
                        (obsolete, computed automatically from TI and
                        simGridAsTiFlag, should be set to None)

        TI:         (1-dimensional array of Img (class)) TI(s) used for the
                        simulation, may contain None entries;
                        it must be compatible with simGridAsTiFlag

        simGridAsTiFlag:
                    (1-dimensional array of 'bool' or None)
                        flag indicating if the simulation grid itself is used
                        as TI, for each TI;
                        if None, an array of False is considered;
                        must be compatible with simGridAsTiFlag

        pdfTI:      ((nTI, nz, ny, nx) array of floats) probability for TI
                        selection, pdf[i] is the "map defined on the SG" of the
                        probability to select the i-th TI, unused if only more
                        than one TI are used (nTI > 1)

        dataImage:  (1-dimensional array of Img (class), or None) data images
                        used as conditioning data (if any), each data image
                        should have the same grid dimensions as those of the SG
                        and its variable name(s) should be included in 'varname';
                        note that the variable names should be distinct, and each
                        data image initializes the corresponding variable in the
                        SG
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
                                ((2, nz, ny, nx) array of floats)
                                    min (homothetyXRatio[0]) and
                                    max (homothetyXRatio[1]) values on the SG
                            else:
                                (1-dimensional array of 2 floats)
                                min and max values
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
                                ((2, nz, ny, nx) array of floats)
                                    min (rotationAzimuth[0]) and
                                    max (rotationAzimuth[1]) values on the SG
                            else:
                                (1-dimensional array of 2 floats)
                                min and max values
                        (unused if rotationUsage == 0)
        rotationDipLocal, rotationDip:
                        as rotationAzimuthLocal and rotationAzimuth, but for
                            the dip angle
        rotationPlungeLocal, rotationPlunge:
                        as rotationAzimuthLocal and rotationAzimuth, but for
                            the plunge angle
        expMax:
            (float) maximal expansion (negative to not check consistency):
                the following is applied for each variable separetely:
                - for variable with distance type set to 0 (see below):
                    * expMax >= 0:
                        if a conditioning data value is not in the set of
                        training image values, an error occurs
                    * expMax < 0:
                        if a conditioning data value is not in the set of
                        training image values, a warning is displayed (no error
                        occurs)
                - for variable with distance type not set to 0 (see below):
                    if relative distance flag is set to 1 (see below), nothing
                    is done, else:
                    * expMax >= 0:
                        maximal accepted expansion of the range of the training
                        image values for covering the conditioning data values:
                        - if conditioning data values are within the range of the
                            training image values: nothing is done
                        - if a conditioning data value is out of the range of the
                            training image values: let
                                new_min_ti = min ( min_cd, min_ti )
                                new_max_ti = max ( max_cd, max_ti )
                            with
                                min_cd, max_cd, the min and max of the
                                    conditioning values,
                                min_ti, max_ti, the min and max of the training
                                    imges values.
                            If new_max_ti-new_min_ti <= (1+expMax)*(ti_max-ti_min)
                            then the training image values are linearly rescaled
                            from [ti_min, ti_max] to [new_ti_min, new_ti_max], and
                            a warning is displayed (no error occurs). Otherwise,
                            an error occurs.
                    * expMax < 0:
                        if a conditioning data value is out of the range of the
                        training image values, a warning is displayed (no error
                        occurs), the training image values are not modified

        normalizingType:
                (string) normalizing type for non categorical variable
                    (distance type not equal to 0), possible strings:
                    'linear', 'uniform', 'normal'

        searchNeighborhoodParameters:
                (1-dimensional array of SearchNeighborhoodParameters (class) of
                    size nv) search neighborhood parameters for each variable
        nneighboringNode:
                (1-dimensional array of ints of size nv) maximal number of
                    neighbors in the search pattern, for each variable
        maxPropInequalityNode:
                (1-dimensional array of doubles of size nv) maximal proportion
                    of nodes with inequality data in the search pattern, for each
                    variable
        neighboringNodeDensity:
                (1-dimensional array of doubles of size nv) density of neighbors
                    in the search pattern, for each variable

        rescalingMode:
                (list of strings of length nv) rescaling mode for each variable,
                    possible strings: 'none', 'min_max', 'mean_length'
        rescalingTargetMin:
                (1-dimensional array of doubles of size nv) target min value,
                    for each variable (used for variable with rescalingMode set
                    to 'min_max')
        rescalingTargetMax:
                (1-dimensional array of doubles of size nv) target max value,
                    for each variable (used for variable with rescalingMode set
                    to 'min_max')
        rescalingTargetMean:
                (1-dimensional array of doubles of size nv) target mean value,
                    for each variable (used for variable with rescalingMode set
                    to 'mean_length')
        rescalingTargetLength:
                (1-dimensional array of doubles of size nv) target length value,
                    for each variable (used for variable with rescalingMode set
                    to 'mean_length')

        relativeDistanceFlag:
                (1-dimensional array of 'bool', of size nv)
                    flag for each variable indicating if relative distance is
                    used (True) or not (False)
        distanceType:
                (list (or 1-dimensional array) of ints or strings of size nv)
                    distance type (between pattern) for each variable; possible
                    values:
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
                    - 'random_hd_distance_pdf': random path set according to
                        distance to conditioning nodes based on pdf,
                        required field 'simPathStrength', see below
                    - 'random_hd_distance_sort': random path set according to
                        distance to conditioning nodes based on sort (with a
                        random noise contribution),
                        required field 'simPathStrength', see below
                    - 'random_hd_distance_sum_pdf': random path set according to
                        sum of distance to conditioning nodes based on pdf,
                        required fields 'simPathPower' and 'simPathStrength',
                        see below
                    - 'random_hd_distance_sum_sort': random path set according to
                        sum of distance to conditioning nodes based on sort (with
                        a random noise contribution),
                        required fields 'simPathPower' and 'simPathStrength',
                        see below
                    - 'unilateral': unilateral path or stratified random path,
                        required field 'simPathUnilateralOrder', see below
        simPathStrength:
                (double) strength in [0,1] attached to distance if simPathType is
                    'random_hd_distance_pdf' or 'random_hd_distance_sort' or
                    'random_hd_distance_sum_pdf' or 'random_hd_distance_sum_sort'
                    (unused otherwise)
        simPathPower:
                (double) power (>0) to which the distance to each conditioning
                    node are elevated, if simPathType is
                    'random_hd_distance_sum_pdf' or 'random_hd_distance_sum_sort'
                    (unused otherwise)
        simPathUnilateralOrder:
                (1-dimesional array of ints), used when simPathType == 'unilateral'
                    - if simType == 'sim_one_by_one': simPathUnilateralOrder is
                        of length 4, example: [0, -2, 1, 0] means that the path
                        will visit all nodes: randomly in xv-sections, with
                        increasing z-coordinate, and then decreasing y-coordinate
                    - if simType == 'sim_variable_vector': simPathUnilateralOrder
                        is of length 3, example: [-1, 0, 2] means that the path
                        will visit all nodes: randomly in y-sections, with
                        decreasing x-coordinate, and then increasing z-coordinate

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

        pyramidDataImage:
            (1-dimensional array of Img (class), or None) data images
                used as conditioning data (if any) in pyramid (in additional
                levels); for each data image:
                    - the variables are identified by their name:
                        the name should be set to <vname>_ind<j>_lev<k>,
                        where <vname> is the name of the "original"
                        variable, <j> is the pyramid index for that variable,
                        and <k> is the level index in {1, ...}
                        (<j> and <k> are written on 3 digits with leading zeros)
                    - the conditioning data values are the (already) normalized
                        values (as used during the simulation in every level)
                    - the grid dimensions (support) of the level in which the data
                        are given are used: the image grid must be compatible
                Note: conditioning data integrated in pyramid may erased
                (replaced) data already set or computed from conditioning data at
                the level one rank finer

        pyramidDataPointSet:
            (1-dimensional array of PointSet (class), or None) point sets
                defining hard data (if any) in pyramid (in additional
                levels); for each point set:
                    - the variables are identified by their name:
                        the name should be set to <vname>_ind<j>_lev<k>,
                        where <vname> is the name of the "original"
                        variable, <j> is the pyramid index for that variable,
                        and <k> is the level index in {1, ...}
                        (<j> and <k> are written on 3 digits with leading zeros)
                    - the conditioning data values are the (already) normalized
                        values (as used during the simulation in every level)
                    - the grid dimensions (support) of the level in which the data
                        are given are used: locations (coordinates) of the points
                        must be given accordingly
                Note: conditioning data integrated in pyramid may erased
                (replaced) data already set or computed from conditioning data at
                the level one rank finer

        tolerance:
                (float) tolerance on the (acceptance) threshold value for
                    flagging nodes (for post-processing)

        npostProcessingPathMax:
                (int) maximal number of post-processing path(s)
                    (0 for no post-processing)

        postProcessingNneighboringNode:
                (1-dimensional array of ints of size nv) maximal number of
                    neighbors in the search pattern, for each variable (for all
                    post-processing paths)

        postProcessingNeighboringNodeDensity:
                (1-dimensional array of doubles of size nv) density of neighbors
                    in the search pattern, for each variable (for all
                    post-processing paths)

        postProcessingDistanceThreshold:
                (1-dimensional array of floats of size nv) distance (acceptance)
                    threshold for each variable (for all post-processing paths)

        postProcessingMaxScanFraction:
                (1-dimensional array of doubles of size nTI) maximal scan
                    fraction of each TI (for all post-processing paths)

        postProcessingTolerance:
                (float) tolerance on the (acceptance) threshold value for
                    flagging nodes (for post-processing) (for all post-processing
                    paths)

        seed:   (int) initial seed
        seedIncrement:
                (int) increment seed

        nrealization:
                (int) number of realization(s)

    Note: in output simulated images (obtained by running DeeSse), the names
        of the output variables are set to <vname>_real<n>, where
            - <vname> is the name of the variable,
            - <n> is the realization index (starting from 0)
            [<n> is written on 5 digits, with leading zeros]
    """

    def __init__(self,
                 simName='deesse_py',
                 nx=1,   ny=1,   nz=1,
                 sx=1.0, sy=1.0, sz=1.0,
                 ox=0.0, oy=0.0, oz=0.0,
                 nv=0, varname=None, outputVarFlag=None,
                 outputPathIndexFlag=False, #outputPathIndexFileName=None,
                 outputErrorFlag=False, #outputErrorFileName=None,
                 outputTiGridNodeIndexFlag=False, #outputTiGridNodeIndexFileName=None,
                 outputTiIndexFlag=False, #outputTiIndexFileName=None,
                 outputReportFlag=False, outputReportFileName=None,
                 nTI=None, TI=None, simGridAsTiFlag=None, pdfTI=None,
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
                 simPathStrength=None,
                 simPathPower=None,
                 simPathUnilateralOrder=None,
                 distanceThreshold=None,
                 softProbability=None,
                 connectivity=None,
                 blockData=None,
                 maxScanFraction=None,
                 pyramidGeneralParameters=None,
                 pyramidParameters=None,
                 pyramidDataImage=None, pyramidDataPointSet=None,
                 tolerance=0.0,
                 npostProcessingPathMax=0,
                 postProcessingNneighboringNode=None,
                 postProcessingNeighboringNodeDensity=None,
                 postProcessingDistanceThreshold=None,
                 postProcessingMaxScanFraction=None,
                 postProcessingTolerance=0.0,
                 seed=1234,
                 seedIncrement=1,
                 nrealization=1):

        fname = 'DeesseInput'

        self.ok = False # flag to "validate" the class [temporary to False]

        # consoleAppFlag
        self.consoleAppFlag = False

        # simulation name
        self.simName = simName

        # grid definition and variable(s)
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)
        self.ox = float(ox)
        self.oy = float(oy)
        self.oz = float(oz)
        self.nv = int(nv)
        if varname is None:
            self.varname = ["V{:d}".format(i) for i in range(nv)]
        else:
            try:
                self.varname = list(np.asarray(varname).reshape(nv))
            except:
                print(f'ERROR ({fname}): field "varname"...')
                return None

        # dimension
        dim = int(nx>1) + int(ny>1) + int(nz>1)

        # outputVarFlag
        if outputVarFlag is None:
            self.outputVarFlag = np.array([True for i in range(nv)], dtype='bool')
        else:
            try:
                self.outputVarFlag = np.asarray(outputVarFlag, dtype='bool').reshape(nv)
            except:
                print(f'ERROR ({fname}): field "outputVarFlag"...')
                return None

        # output maps
        self.outputPathIndexFlag = outputPathIndexFlag
        # self.outputPathIndexFileName = None # no output file!

        self.outputErrorFlag = outputErrorFlag
        # self.outputErrorFileName = None # no output file!

        self.outputTiGridNodeIndexFlag = outputTiGridNodeIndexFlag
        # self.outputTiGridNodeIndexFileName = None # no output file!

        self.outputTiIndexFlag = outputTiIndexFlag
        # self.outputTiIndexFileName = None # no output file!

        # report
        self.outputReportFlag = outputReportFlag
        if outputReportFileName is None:
            self.outputReportFileName ='ds.log'
        else:
            self.outputReportFileName = outputReportFileName

        # TI, simGridAsTiFlag, nTI
        if TI is None and simGridAsTiFlag is None:
            print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag" (both None)...')
            return None

        if TI is not None:
            self.TI = np.asarray(TI).reshape(-1)

        if simGridAsTiFlag is not None:
            self.simGridAsTiFlag = np.asarray(simGridAsTiFlag, dtype='bool').reshape(-1)

        if TI is None:
            self.TI = np.array([None for i in range(len(self.simGridAsTiFlag))], dtype=object)

        if simGridAsTiFlag is None:
            self.simGridAsTiFlag = np.array([False for i in range(len(self.TI))], dtype='bool') # set dtype='bool' in case of len(self.TI)=0

        if len(self.TI) != len(self.simGridAsTiFlag):
            print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag" (not same length)...')
            return None

        for f, t in zip(self.simGridAsTiFlag, self.TI):
            if (not f and t is None) or (f and t is not None):
                print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag"...')
                return None

        if nTI is not None and nTI != len(self.TI):
            print(f'ERROR ({fname}): invalid "nTI"...')
            return None

        nTI = len(self.TI)
        self.nTI = nTI

        # pdfTI
        if nTI <= 1:
            self.pdfTI = None
        else:
            if pdfTI is None:
                p = 1./nTI
                self.pdfTI = np.repeat(p, nTI*nx*ny*nz).reshape(nTI, nz, ny, nx)
            else:
                try:
                    self.pdfTI = np.asarray(pdfTI, dtype=float).reshape(nTI, nz, ny, nx)
                except:
                    print(f'ERROR ({fname}): field "pdfTI"...')
                    return None

        # conditioning data image
        if dataImage is None:
            self.dataImage = None
        else:
            self.dataImage = np.asarray(dataImage).reshape(-1)

        # conditioning point set
        if dataPointSet is None:
            self.dataPointSet = None
        else:
            self.dataPointSet = np.asarray(dataPointSet).reshape(-1)

        # mask
        if mask is None:
            self.mask = None
        else:
            try:
                self.mask = np.asarray(mask).reshape(nz, ny, nx)
            except:
                print(f'ERROR ({fname}): field "mask"...')
                return None

        # homothety
        if homothetyUsage == 1:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None

        elif homothetyUsage == 2:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None

        elif homothetyUsage == 0:
            self.homothetyXRatio = None
            self.homothetyYRatio = None
            self.homothetyZRatio = None

        else:
            print(f'ERROR ({fname}): invalid homothetyUsage')
            return None

        self.homothetyUsage = homothetyUsage
        self.homothetyXLocal = homothetyXLocal
        self.homothetyYLocal = homothetyYLocal
        self.homothetyZLocal = homothetyZLocal

        # rotation
        if rotationUsage == 1:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None

        elif rotationUsage == 2:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0., 0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0., 0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0., 0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None

        elif rotationUsage == 0:
            self.rotationAzimuth = None
            self.rotationDip = None
            self.rotationPlunge = None

        else:
            print(f'ERROR ({fname}): invalid rotationUsage')
            return None

        self.rotationUsage = rotationUsage
        self.rotationAzimuthLocal = rotationAzimuthLocal
        self.rotationDipLocal = rotationDipLocal
        self.rotationPlungeLocal = rotationPlungeLocal

        # expMax
        self.expMax = expMax

        # normalizing type
        # if normalizingType not in ('linear', 'uniform', 'normal'):
        #     print('ERRROR: (DeesseInput) field "normalizingType"')
        #     return None

        self.normalizingType = normalizingType

        # search neighborhood, number of neighbors, ...
        if searchNeighborhoodParameters is None:
            self.searchNeighborhoodParameters = np.array([SearchNeighborhoodParameters() for i in range(nv)])
        else:
            try:
                self.searchNeighborhoodParameters = np.asarray(searchNeighborhoodParameters).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "searchNeighborhoodParameters"...')
                return None

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
                print(f'ERROR ({fname}): field "nneighboringNode"...')
                return None

        if maxPropInequalityNode is None:
            self.maxPropInequalityNode = np.array([0.25 for i in range(nv)])
        else:
            try:
                self.maxPropInequalityNode = np.asarray(maxPropInequalityNode).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "maxPropInequalityNode"...')
                return None

        if neighboringNodeDensity is None:
            self.neighboringNodeDensity = np.array([1. for i in range(nv)])
        else:
            try:
                self.neighboringNodeDensity = np.asarray(neighboringNodeDensity, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "neighboringNodeDensity"...')
                return None

        # rescaling
        if rescalingMode is None:
            self.rescalingMode = ['none' for i in range(nv)]
        else:
            try:
                self.rescalingMode = list(np.asarray(rescalingMode).reshape(nv))
            except:
                print(f'ERROR ({fname}): field "rescalingMode"...')
                return None

        if rescalingTargetMin is None:
            self.rescalingTargetMin = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMin = np.asarray(rescalingTargetMin, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMin"...')
                return None

        if rescalingTargetMax is None:
            self.rescalingTargetMax = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMax = np.asarray(rescalingTargetMax, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMax"...')
                return None

        if rescalingTargetMean is None:
            self.rescalingTargetMean = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMean = np.asarray(rescalingTargetMean, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMean"...')
                return None

        if rescalingTargetLength is None:
            self.rescalingTargetLength = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetLength = np.asarray(rescalingTargetLength, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetLength"...')
                return None

        # distance, ...
        if relativeDistanceFlag is None:
            self.relativeDistanceFlag = np.array([False for i in range(nv)], dtype='bool') # set dtype='bool' in case of nv=0
        else:
            try:
                self.relativeDistanceFlag = np.asarray(relativeDistanceFlag, dtype='bool').reshape(nv)
            except:
                print(f'ERROR ({fname}): field "relativeDistanceFlag"...')
                return None

        if powerLpDistance is None:
            self.powerLpDistance = np.array([1. for i in range(nv)])
        else:
            try:
                self.powerLpDistance = np.asarray(powerLpDistance, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "powerLpDistance"...')
                return None

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
                            print(f'ERROR ({fname}): field "distanceType"...')
                            return None
                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "distanceType"...')
                return None

        # conditioning weight
        if conditioningWeightFactor is None:
            self.conditioningWeightFactor = np.array([1. for i in range(nv)])
        else:
            try:
                self.conditioningWeightFactor = np.asarray(conditioningWeightFactor, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "conditioningWeightFactor"...')
                return None

        # simulation type and simulation path type
        if simType not in ('sim_one_by_one', 'sim_variable_vector'):
            print('ERRROR: (DeesseInput) field "simType"...')
            return None

        self.simType = simType

        if simPathType not in (
                'random',
                'random_hd_distance_pdf', 'random_hd_distance_sort',
                'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort',
                'unilateral'):
            print('ERRROR: (DeesseInput) field "simPathType"...')
            return None

        self.simPathType = simPathType

        if simPathStrength is None:
            simPathStrength = 0.5
        if simPathPower is None:
            simPathPower = 2.0

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
                    print(f'ERROR ({fname}): field "simPathUnilateralOrder"...')
                    return None
        else:
            self.simPathUnilateralOrder = None

        # distance threshold
        if distanceThreshold is None:
            self.distanceThreshold = np.array([0.05 for i in range(nv)])
        else:
            try:
                self.distanceThreshold = np.asarray(distanceThreshold, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "distanceThreshold"...')
                return None

        # soft probability
        if softProbability is None:
            self.softProbability = np.array([SoftProbability(probabilityConstraintUsage=0) for i in range(nv)])
        else:
            try:
                self.softProbability = np.asarray(softProbability).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "softProbability"...')
                return None

        # connectivity
        if connectivity is None:
            self.connectivity = np.array([Connectivity(connectivityConstraintUsage=0) for i in range(nv)])
        else:
            try:
                self.connectivity = np.asarray(connectivity).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "connectivity"...')
                return None

        # block data
        if blockData is None:
            self.blockData = np.array([BlockData(blockDataUsage=0) for i in range(nv)])
        else:
            try:
                self.blockData = np.asarray(blockData).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "blockData"...')
                return None

        # maximal scan fraction
        if maxScanFraction is None:
            if dim == 3: # 3D simulation
                nf = 10000
            else:
                nf = 5000

            self.maxScanFraction = np.array([min(max(nf/self.TI[i].nxyz(), deesse.MPDS_MIN_MAXSCANFRACTION), deesse.MPDS_MAX_MAXSCANFRACTION) for i in range(nTI)])
        else:
            try:
                self.maxScanFraction = np.asarray(maxScanFraction).reshape(nTI)
            except:
                print(f'ERROR ({fname}): field "maxScanFraction"...')
                return None

        # pyramids
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
                print(f'ERROR ({fname}): field "pyramidParameters"...')
                return None

        if pyramidDataImage is None:
            self.pyramidDataImage = None
        else:
            self.pyramidDataImage = np.asarray(pyramidDataImage).reshape(-1)

        if pyramidDataPointSet is None:
            self.pyramidDataPointSet = None
        else:
            self.pyramidDataPointSet = np.asarray(pyramidDataPointSet).reshape(-1)

        # tolerance and post-processing
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
                print(f'ERROR ({fname}): field "postProcessingNneighboringNode"...')
                return None

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
                print(f'ERROR ({fname}): field "postProcessingNeighboringNodeDensity"...')
                return None

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
                print(f'ERROR ({fname}): field "postProcessingDistanceThreshold"...')
                return None

        if postProcessingMaxScanFraction is None:
            self.postProcessingMaxScanFraction = np.array([min(deesse.MPDS_POST_PROCESSING_MAX_SCAN_FRACTION_DEFAULT, self.maxScanFraction[i]) for i in range(nTI)], dtype=float)

        else:
            try:
                self.postProcessingMaxScanFraction = np.asarray(postProcessingMaxScanFraction, dtype=float).reshape(nTI)
            except:
                print(f'ERROR ({fname}): field "postProcessingMaxScanFraction"...')
                return None

        self.postProcessingTolerance = postProcessingTolerance

        # seed, ...
        if seed is None:
            seed = np.random.randint(1,1000000)
        self.seed = seed
        self.seedIncrement = seedIncrement

        # number of realization(s)
        self.nrealization = nrealization

        self.ok = True # flag to "validate" the class

    # ------------------------------------------------------------------------
    # def __str__(self):
    def __repr__(self):
        out = '*** DeesseInput object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ----------------------------------------------------------------------------
def img_py2C(im):
    """
    Converts an image from python to C.

    :param im:      (Img class) image (python class)
    :return im_c:   (MPDS_IMAGE *) image converted (C struct)
    """

    fname = 'img_py2C'

    im_c = deesse.malloc_MPDS_IMAGE()
    deesse.MPDSInitImage(im_c)

    err = deesse.MPDSMallocImage(im_c, im.nxyz(), im.nv)
    if err:
        print(f'ERROR ({fname}): can not convert image from python to C')
        return None

    im_c.grid.nx = im.nx
    im_c.grid.ny = im.ny
    im_c.grid.nz = im.nz

    im_c.grid.sx = im.sx
    im_c.grid.sy = im.sy
    im_c.grid.sz = im.sz

    im_c.grid.ox = im.ox
    im_c.grid.oy = im.oy
    im_c.grid.oz = im.oz

    im_c.grid.nxy = im.nxy()
    im_c.grid.nxyz = im.nxyz()

    im_c.nvar = im.nv

    im_c.nxyzv = im.nxyz() * im.nv

    for i in range(im.nv):
        deesse.mpds_set_varname(im_c.varName, i, im.varname[i])
        # deesse.charp_array_setitem(im_c.varName, i, im.varname[i]) # does not work!

    v = im.val.reshape(-1)
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
    :return im:      (Img class) image converted (python class)
    """

    nxyz = im_c.grid.nx * im_c.grid.ny * im_c.grid.nz
    nxyzv = nxyz * im_c.nvar

    varname = [deesse.mpds_get_varname(im_c.varName, i) for i in range(im_c.nvar)]
    # varname = [deesse.charp_array_getitem(im_c.varName, i) for i in range(im_c.nvar)] # also works

    v = np.zeros(nxyzv)
    deesse.mpds_get_array_from_real_vector(im_c.var, 0, v)

    im = Img(nx=im_c.grid.nx, ny=im_c.grid.ny, nz=im_c.grid.nz,
             sx=im_c.grid.sx, sy=im_c.grid.sy, sz=im_c.grid.sz,
             ox=im_c.grid.ox, oy=im_c.grid.oy, oz=im_c.grid.oz,
             nv=im_c.nvar, val=v, varname=varname)

    np.putmask(im.val, im.val == deesse.MPDS_MISSING_VALUE, np.nan)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def ps_py2C(ps):
    """
    Converts a point set from python to C.

    :param ps:      (PointSet class) point set (python class)
    :return ps_c:   (MPDS_POINTSET *) point set converted (C struct)
    """

    fname = 'ps_py2C'

    if ps.nv < 4:
        print(f'ERROR ({fname}): point set (python) have less than 4 variables')
        return None

    nvar = ps.nv - 3

    ps_c = deesse.malloc_MPDS_POINTSET()
    deesse.MPDSInitPointSet(ps_c)

    err = deesse.MPDSMallocPointSet(ps_c, ps.npt, nvar)
    if err:
        print(f'ERROR ({fname}): can not convert point set from python to C')
        return None

    ps_c.npoint = ps.npt
    ps_c.nvar = nvar

    for i in range(nvar):
        deesse.mpds_set_varname(ps_c.varName, i, ps.varname[i+3])

    deesse.mpds_set_real_vector_from_array(ps_c.x, 0, ps.val[0].reshape(-1))
    deesse.mpds_set_real_vector_from_array(ps_c.y, 0, ps.val[1].reshape(-1))
    deesse.mpds_set_real_vector_from_array(ps_c.z, 0, ps.val[2].reshape(-1))

    v = ps.val[3:].reshape(-1)
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
    :return ps:     (PointSet class) point set converted (python class)
    """

    varname = ['X', 'Y', 'Z'] + [deesse.mpds_get_varname(ps_c.varName, i) for i in range(ps_c.nvar)]
    # varname = ['X', 'Y', 'Z'] + [deesse.charp_array_getitem(ps_c.varName, i) for i in range(ps_c.nvar)] # also works

    v = np.zeros(ps_c.npoint*ps_c.nvar)
    deesse.mpds_get_array_from_real_vector(ps_c.var, 0, v)

    # coord = np.zeros(ps_c.npoint)
    # deesse.mpds_get_array_from_real_vector(ps_c.z, 0, coord)
    # v = np.hstack(coord,v)
    # deesse.mpds_get_array_from_real_vector(ps_c.y, 0, coord)
    # v = np.hstack(coord,v)
    # deesse.mpds_get_array_from_real_vector(ps_c.x, 0, coord)
    # v = np.hstack(coord,v)

    cx = np.zeros(ps_c.npoint)
    cy = np.zeros(ps_c.npoint)
    cz = np.zeros(ps_c.npoint)
    deesse.mpds_get_array_from_real_vector(ps_c.x, 0, cx)
    deesse.mpds_get_array_from_real_vector(ps_c.y, 0, cy)
    deesse.mpds_get_array_from_real_vector(ps_c.z, 0, cz)
    v = np.hstack((cx, cy, cz, v))

    ps = PointSet(npt=ps_c.npoint,
                     nv=ps_c.nvar+3, val=v, varname=varname)

    np.putmask(ps.val, ps.val == deesse.MPDS_MISSING_VALUE, np.nan)

    return ps
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
def classOfValues2classInterval(classOfValues):
    """
    Converts classOfValues (C) to classInterval (python).

    :param cv:  (MPDS_CLASSOFVALUES *) corresponding structure in C

    :return classInterval:
                    (list of nclass 2-dimensional array of floats with 2 columns)
                        definition of the classes of values by intervals,
                        classInterval[i] is a (n_i, 2) array a, defining the
                        i-th class as the union of intervals:
                            [a[0,0],a[0,1][ U ... [a[n_i-1,0],a[n_i-1,1][
    """

    nclass = classOfValues.nclass

    ninterval = np.zeros(nclass, dtype='intc')
    deesse.mpds_get_array_from_int_vector(classOfValues.ninterval, 0, ninterval)
    ninterval = ninterval.astype('int')

    classInterval = nclass*[None]
    for i in range(nclass):
        intervalInf = np.zeros(ninterval[i])
        rptr = deesse.realp_array_getitem(classOfValues.intervalInf, i)
        deesse.mpds_get_array_from_real_vector(rptr, 0, intervalInf)
        intervalSup = np.zeros(ninterval[i])
        rptr = deesse.realp_array_getitem(classOfValues.intervalSup, i)
        deesse.mpds_get_array_from_real_vector(rptr, 0, intervalSup)
        classInterval[i] = np.array((intervalInf, intervalSup)).T

    return classInterval
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def search_neighborhood_parameters_py2C(sn):
    """
    Converts search neighborhood parameters from python to C.

    :param sn:      (SearchNeighborhoodParameters class) - python
    :return sn_c:   (MPDS_SEARCHNEIGHBORHOODPARAMETERS *) - C
    """

    fname = 'search_neighborhood_parameters_py2C'

    sn_c = deesse.malloc_MPDS_SEARCHNEIGHBORHOODPARAMETERS()
    deesse.MPDSInitSearchNeighborhoodParameters(sn_c)

    radiusMode_dict = {
        'large_default'    : deesse.SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT,
        'ti_range_default' : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT,
        'ti_range'         : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE,
        'ti_range_xy'      : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY,
        'ti_range_xz'      : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ,
        'ti_range_yz'      : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ,
        'ti_range_xyz'     : deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ,
        'manual'           : deesse.SEARCHNEIGHBORHOOD_RADIUS_MANUAL
    }
    try:
        sn_c.radiusMode = radiusMode_dict[sn.radiusMode]
    except:
        print(f'ERROR ({fname}): radius mode (search neighborhood parameters) unknown')
        return None

    sn_c.rx = sn.rx
    sn_c.ry = sn.ry
    sn_c.rz = sn.rz

    anisotropyRatioMode_dict = {
        'one'        : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE,
        'radius'     : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS,
        'radius_xy'  : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY,
        'radius_xz'  : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ,
        'radius_yz'  : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ,
        'radius_xyz' : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ,
        'manual'     : deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_MANUAL
    }
    try:
        sn_c.anisotropyRatioMode = anisotropyRatioMode_dict[sn.anisotropyRatioMode]
    except:
        print(f'ERROR ({fname}): anisotropy ratio mode (search neighborhood parameters) unknown')
        return None

    sn_c.ax = sn.ax
    sn_c.ay = sn.ay
    sn_c.az = sn.az
    sn_c.angle1 = sn.angle1
    sn_c.angle2 = sn.angle2
    sn_c.angle3 = sn.angle3
    if sn_c.angle1 != 0 or sn_c.angle2 != 0 or sn_c.angle3 != 0:
        sn_c.rotationFlag = deesse.TRUE
    else:
        sn_c.rotationFlag = deesse.FALSE
    sn_c.power = sn.power

    return sn_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def search_neighborhood_parameters_C2py(sn_c):
    """
    Converts search neighborhood parameters from python to C.

    :param sn_c:    (MPDS_SEARCHNEIGHBORHOODPARAMETERS *) - C
    :return sn:     (SearchNeighborhoodParameters class) - python
    """

    fname = 'search_neighborhood_parameters_C2py'

    radiusMode_dict = {
        deesse.SEARCHNEIGHBORHOOD_RADIUS_LARGE_DEFAULT    : 'large_default',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_DEFAULT : 'ti_range_default',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE         : 'ti_range',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XY      : 'ti_range_xy',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XZ      : 'ti_range_xz',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_YZ      : 'ti_range_yz',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_TI_RANGE_XYZ     : 'ti_range_xyz',
        deesse.SEARCHNEIGHBORHOOD_RADIUS_MANUAL           : 'manual'
    }
    try:
        radiusMode = radiusMode_dict[sn_c.radiusMode]
    except:
        print(f'ERROR ({fname}): radius mode (search neighborhood parameters) unknown')
        return None

    rx = sn_c.rx
    ry = sn_c.ry
    rz = sn_c.rz

    anisotropyRatioMode_dict = {
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_ONE        : 'one',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS     : 'radius',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XY  : 'radius_xy',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XZ  : 'radius_xz',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_YZ  : 'radius_yz',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_RADIUS_XYZ : 'radius_xyz',
        deesse.SEARCHNEIGHBORHOOD_ANISOTROPY_RATIO_MANUAL     : 'manual'
    }
    try:
        anisotropyRatioMode = anisotropyRatioMode_dict[sn_c.anisotropyRatioMode]
    except:
        print(f'ERROR ({fname}): anisotropy ratio mode (search neighborhood parameters) unknown')
        return None

    ax = sn_c.ax
    ay = sn_c.ay
    az = sn_c.az
    angle1 = sn_c.angle1
    angle2 = sn_c.angle2
    angle3 = sn_c.angle3
    power = sn_c.power

    sn = SearchNeighborhoodParameters(
        radiusMode=radiusMode,
        rx=rx, ry=ry, rz=rz,
        anisotropyRatioMode=anisotropyRatioMode,
        ax=ax, ay=ay, az=az,
        angle1=angle1, angle2=angle2, angle3=angle3,
        power=power
    )

    return sn
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def set_simAndPathParameters_C(
        simType,
        simPathType,
        simPathStrength,
        simPathPower,
        simPathUnilateralOrder):
    """
    Set simAndPathParameters (C struct) from relevant parameters (python).

    :param simType, simPathType, simPathStrength, simPathUnilateralOrder:
                    relevant parameters - python

    :return sapp_c:   (MPDS_SIMANDPATHPARAMETERS *) - C
    """

    fname = 'set_simAndPathParameters_C'

    sapp_c = deesse.malloc_MPDS_SIMANDPATHPARAMETERS()
    deesse.MPDSInitSimAndPathParameters(sapp_c)

    if simType == 'sim_one_by_one':
        sapp_c.simType = deesse.SIM_ONE_BY_ONE
    elif simType == 'sim_variable_vector':
        sapp_c.simType = deesse.SIM_VARIABLE_VECTOR
    else:
        print(f'ERROR ({fname}): simulation type unknown')
        return None

    if simPathType == 'random':
        sapp_c.pathType = deesse.PATH_RANDOM
    elif simPathType == 'random_hd_distance_pdf':
        sapp_c.pathType = deesse.PATH_RANDOM_HD_DISTANCE_PDF
        sapp_c.strength = simPathStrength
    elif simPathType == 'random_hd_distance_sort':
        sapp_c.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SORT
        sapp_c.strength = simPathStrength
    elif simPathType == 'random_hd_distance_sum_pdf':
        sapp_c.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SUM_PDF
        sapp_c.pow = simPathPower
        sapp_c.strength = simPathStrength
    elif simPathType == 'random_hd_distance_sum_sort':
        sapp_c.pathType = deesse.PATH_RANDOM_HD_DISTANCE_SUM_SORT
        sapp_c.pow = simPathPower
        sapp_c.strength = simPathStrength
    elif simPathType == 'unilateral':
        sapp_c.pathType = deesse.PATH_UNILATERAL
        sapp_c.unilateralOrderLength = len(simPathUnilateralOrder)
        sapp_c.unilateralOrder = deesse.new_int_array(len(simPathUnilateralOrder))
        deesse.mpds_set_int_vector_from_array(
            sapp_c.unilateralOrder, 0,
            np.asarray(simPathUnilateralOrder, dtype='intc').reshape(len(simPathUnilateralOrder)))
    else:
        print(f'ERROR ({fname}): simulation path type unknown')
        return None

    return sapp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def softProbability_py2C(
        sp,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz):
    """
    Converts soft probability parameters from python to C.

    :param sp:          (SoftProbability class) soft probability parameters
                            (python)
    :param nx, ny, nz:  (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz:  (floats) cell size in each direction
    :param ox, oy, oz:  (floats) origin of the SG (bottom-lower-left corner)
    :param nv:          (int) number of variable(s) / attribute(s)

    :return sp_c:       (MPDS_SOFTPROBABILITY *) corresponding parameters
                            (C struct)
    """

    #fname = 'softProbability_py2C'

    sp_c = deesse.malloc_MPDS_SOFTPROBABILITY()
    deesse.MPDSInitSoftProbability(sp_c)

    # ... probabilityConstraintUsage
    sp_c.probabilityConstraintUsage = sp.probabilityConstraintUsage
    if sp.probabilityConstraintUsage == 0:
        return sp_c

    # ... classOfValues
    sp_c.classOfValues = classInterval2classOfValues(sp.classInterval)

    if sp.probabilityConstraintUsage == 1:
        # ... globalPdf
        sp_c.globalPdf = deesse.new_real_array(sp.nclass)
        deesse.mpds_set_real_vector_from_array(
            sp_c.globalPdf, 0,
            np.asarray(sp.globalPdf).reshape(sp.nclass))

    elif sp.probabilityConstraintUsage == 2 or sp.probabilityConstraintUsage == 3:
        # ... localPdf
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=sp.nclass, val=sp.localPdf)
        sp_c.localPdfImage = img_py2C(im)

    if sp.probabilityConstraintUsage == 2:
        # ... localPdfSupportRadius
        sp_c.localPdfSupportRadius = deesse.new_real_array(1)
        deesse.mpds_set_real_vector_from_array(
            sp_c.localPdfSupportRadius, 0,
            np.asarray(sp.localPdfSupportRadius).reshape(1))

        # ... localCurrentPdfComputation
        sp_c.localCurrentPdfComputation = sp.localCurrentPdfComputation

    if sp.probabilityConstraintUsage == 1 or sp.probabilityConstraintUsage == 2:
        # ... comparingPdfMethod
        sp_c.comparingPdfMethod = sp.comparingPdfMethod

        # ... probabilityConstraintThresholdType
        sp_c.probabilityConstraintThresholdType = sp.probabilityConstraintThresholdType

        # ... constantThreshold
        sp_c.constantThreshold = sp.constantThreshold

        if sp.probabilityConstraintThresholdType == 1:
            # ... dynamicThresholdParameters
            sp_c.dynamicThresholdParameters = deesse.new_real_array(7)
            deesse.mpds_set_real_vector_from_array(
                sp_c.dynamicThresholdParameters, 0,
                np.asarray(sp.dynamicThresholdParameters).reshape(7))

    if sp.probabilityConstraintUsage == 3:
        # ... rejectionMode
        sp_c.rejectionMode = sp.rejectionMode

    # ... deactivationDistance
    sp_c.deactivationDistance = sp.deactivationDistance

    return sp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def softProbability_C2py(sp_c):
    """
    Converts soft probability parameters from C to python.

    :param sp_c:       (MPDS_SOFTPROBABILITY *) corresponding parameters
                            (C struct)
    :return sp:         (SoftProbability class) soft probability parameters
                            (python)
    """

    # ... probabilityConstraintUsage
    probabilityConstraintUsage = sp_c.probabilityConstraintUsage
    if probabilityConstraintUsage == 0:
        sp = SoftProbability()
        return sp

    # default parameters
    nclass = 0
    classInterval = None
    globalPdf = None
    localPdf = None
    localPdfSupportRadius = 12.0
    localCurrentPdfComputation = 0
    comparingPdfMethod = 5
    rejectionMode = 0
    deactivationDistance = 4.0
    probabilityConstraintThresholdType = 0
    constantThreshold = 1.e-3
    dynamicThresholdParameters = None

    # ... classOfValues
    classInterval = classOfValues2classInterval(sp_c.classOfValues)
    nclass = len(classInterval)

    if probabilityConstraintUsage == 1:
        # ... globalPdf
        globalPdf = np.zeros(nclass, dtype=float)
        deesse.mpds_get_array_from_real_vector(sp_c.globalPdf, 0, globalPdf)

    elif probabilityConstraintUsage == 2 or probabilityConstraintUsage == 3:
        # ... localPdf
        im = img_C2py(sp_c.localPdfImage)
        localPdf = im.val

    if probabilityConstraintUsage == 2:
        # ... localPdfSupportRadius
        v = np.zeros(1)
        deesse.mpds_get_array_from_real_vector(sp_c.localPdfSupportRadius, 0, v)
        localPdfSupportRadius = v[0]

        # ... localCurrentPdfComputation
        localCurrentPdfComputation = sp_c.localCurrentPdfComputation

    if probabilityConstraintUsage == 1 or probabilityConstraintUsage == 2:
        # ... comparingPdfMethod
        comparingPdfMethod = sp_c.comparingPdfMethod

        # ... probabilityConstraintThresholdType
        probabilityConstraintThresholdType = sp_c.probabilityConstraintThresholdType

        # ... constantThreshold
        constantThreshold = sp_c.constantThreshold

        if probabilityConstraintThresholdType == 1:
            # ... dynamicThresholdParameters
            dynamicThresholdParameters = np.zeros(7, dtype=float)
            deesse.mpds_get_array_from_real_vector(sp_c.dynamicThresholdParameters, 0, dynamicThresholdParameters)

    if probabilityConstraintUsage == 3:
        # ... rejectionMode
        rejectionMode = sp_c.rejectionMode

    # ... deactivationDistance
    deactivationDistance = sp_c.deactivationDistance

    sp = SoftProbability(
        probabilityConstraintUsage=probabilityConstraintUsage,
        nclass=nclass,
        classInterval=classInterval,
        globalPdf=globalPdf,
        localPdf=localPdf,
        localPdfSupportRadius=localPdfSupportRadius,
        localCurrentPdfComputation=localCurrentPdfComputation,
        comparingPdfMethod=comparingPdfMethod,
        rejectionMode=rejectionMode,
        deactivationDistance=deactivationDistance,
        probabilityConstraintThresholdType=probabilityConstraintThresholdType,
        constantThreshold=constantThreshold,
        dynamicThresholdParameters=dynamicThresholdParameters
    )

    return sp
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def connectivity_py2C(co):
    """
    Converts connectivity parameters from python to C.

    :param co:      (Connectivity class) connectivity parameters (python)
    :return co_c:   (MPDS_CONNECTIVITY *) corresponding parameters (C struct)
    """

    fname = 'connectivity_py2C'

    co_c = deesse.malloc_MPDS_CONNECTIVITY()
    deesse.MPDSInitConnectivity(co_c)

    # ... connectivityConstraintUsage
    co_c.connectivityConstraintUsage = co.connectivityConstraintUsage
    if co.connectivityConstraintUsage == 0:
        return co_c

    # ... connectivityType
    connectivityType_dict = {
        'connect_face'             : deesse.CONNECT_FACE,
        'connect_face_edge'        : deesse.CONNECT_FACE_EDGE,
        'connect_face_edge_corner' : deesse.CONNECT_FACE_EDGE_CORNER
    }
    try:
        co_c.connectivityType = connectivityType_dict[co.connectivityType]
    except:
        print(f'ERROR ({fname}): connectivity type unknown')
        return None

    # ... varName
    deesse.mpds_allocate_and_set_connectivity_varname(co_c, co.varname)

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

    return co_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def connectivity_C2py(co_c):
    """
    Converts connectivity parameters from C to python.

    :param  co_c:   (MPDS_CONNECTIVITY *) corresponding parameters (C struct)
    :return co:     (Connectivity class) connectivity parameters (python)
    """

    fname = 'connectivity_C2py'

    # ... connectivityConstraintUsage
    connectivityConstraintUsage = co_c.connectivityConstraintUsage
    if connectivityConstraintUsage == 0:
        co = Connectivity()
        return co

    # default parameters
    connectivityType = 'connect_face'
    nclass = 0
    classInterval = None
    varname = ''
    tiAsRefFlag = True
    refConnectivityImage = None
    refConnectivityVarIndex = 0
    deactivationDistance = 0.0
    threshold = 0.01

    # ... connectivityType
    connectivityType_dict = {
        deesse.CONNECT_FACE             : 'connect_face',
        deesse.CONNECT_FACE_EDGE        : 'connect_face_edge',
        deesse.CONNECT_FACE_EDGE_CORNER : 'connect_face_edge_corner'
    }
    try:
        connectivityType = connectivityType_dict[co_c.connectivityType]
    except:
        print(f'ERROR ({fname}): connectivity type unknown')
        return None

    # ... varName
    varname = co_c.varName

    # ... classInterval
    classInterval = classOfValues2classInterval(co_c.classOfValues)
    nclass = len(classInterval)

    # ... tiAsRefFlag
    tiAsRefFlag = bool(int.from_bytes(co_c.tiAsRefFlag.encode('utf-8'), byteorder='big'))

    if not tiAsRefFlag:
        # ... refConnectivityImage
        refConnectivityImage = img_C2py(co_c.refConnectivityImage)
        refConnectivityVarIndex = 0

    # ... deactivationDistance
    deactivationDistance = co_c.deactivationDistance

    # ... threshold
    threshold = co_c.threshold

    co = Connectivity(
        connectivityConstraintUsage=connectivityConstraintUsage,
        connectivityType=connectivityType,
        nclass=nclass,
        classInterval=classInterval,
        varname=varname,
        tiAsRefFlag=tiAsRefFlag,
        refConnectivityImage=refConnectivityImage,
        refConnectivityVarIndex=refConnectivityVarIndex,
        deactivationDistance=deactivationDistance,
        threshold=threshold
    )

    return co
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def blockData_py2C(bd):
    """
    Converts block data parameters from python to C.

    :param bd:      (BlockData class) block data parameters (python)
    :return bd_c:   (MPDS_BLOCKDATA *) corresponding parameters (C struct)
    """

    bd_c = deesse.malloc_MPDS_BLOCKDATA()
    deesse.MPDSInitBlockData(bd_c)

    # ... blockDataUsage
    bd_c.blockDataUsage = bd.blockDataUsage
    if bd.blockDataUsage == 0:
        return bd_c

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

    return bd_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def blockData_C2py(bd_c):
    """
    Converts block data parameters from C to python.

    :param bd_c:    (MPDS_BLOCKDATA *) corresponding parameters (C struct)
    :return bd:     (BlockData class) block data parameters (python)
    """

    # ... blockDataUsage
    blockDataUsage = bd_c.blockDataUsage
    if blockDataUsage == 0:
        bd = BlockData()
        return bd

    # default parameters
    nblock = 0
    nodeIndex = None
    value = None
    tolerance = None
    activatePropMin = None
    activatePropMax = None

    # ... nblock
    nblock = bd_c.nblock

    # ... nnode, nodeIndex (ix, iy, iz)
    nnode = np.zeros(nblock, dtype='intc')
    deesse.mpds_get_array_from_int_vector(bd_c.nnode, 0, nnode)
    nnode = nnode.astype('int')

    nodeIndex = nblock*[None]
    for i in range(nblock):
        ix = np.zeros(nnode[i], dtype='intc')
        iptr = deesse.intp_array_getitem(bd_c.ix, i)
        deesse.mpds_get_array_from_int_vector(iptr, 0, ix)
        ix = ix.astype('int')

        iy = np.zeros(nnode[i], dtype='intc')
        iptr = deesse.intp_array_getitem(bd_c.iy, i)
        deesse.mpds_get_array_from_int_vector(iptr, 0, iy)
        iy = iy.astype('int')

        iz = np.zeros(nnode[i], dtype='intc')
        iptr = deesse.intp_array_getitem(bd_c.iz, i)
        deesse.mpds_get_array_from_int_vector(iptr, 0, iz)
        iz = iz.astype('int')

        nodeIndex[i] = np.array((ix, iy, iz)).T

    # ... value
    value = np.zeros(nblock, dtype=float)
    deesse.mpds_get_array_from_real_vector(bd_c.value, 0, value)

    # ... tolerance
    tolerance = np.zeros(nblock, dtype=float)
    deesse.mpds_get_array_from_real_vector(bd_c.tolerance, 0, tolerance)

    # ... activatePropMin
    activatePropMin = np.zeros(nblock, dtype=float)
    deesse.mpds_get_array_from_real_vector(bd_c.activatePropMin, 0, activatePropMin)

    # ... activatePropMax
    activatePropMax = np.zeros(nblock, dtype=float)
    deesse.mpds_get_array_from_real_vector(bd_c.activatePropMax, 0, activatePropMax)

    bd = BlockData(
         blockDataUsage=blockDataUsage,
         nblock=nblock,
         nodeIndex=nodeIndex,
         value=value,
         tolerance=tolerance,
         activatePropMin=activatePropMin,
         activatePropMax=activatePropMax
    )

    return bd
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidGeneralParameters_py2C(pgp):
    """
    Converts pyramid general parameters from python to C.

    :param pgp:     (PyramidGeneralParameters class) pyramid general parameters
                        (python)
    :return pgp_c:  (MPDS_PYRAMIDGENERALPARAMETERS *) corresponding parameters
                        (C struct)
    """

    pgp_c = deesse.malloc_MPDS_PYRAMIDGENERALPARAMETERS()
    deesse.MPDSInitPyramidGeneralParameters(pgp_c)

    # ... npyramidLevel
    nl = int(pgp.npyramidLevel)
    pgp_c.npyramidLevel = nl

    # ... pyramidSimulationMode
    pyramidSimulationMode_dict = {
        'hierarchical'                 : deesse.PYRAMID_SIM_HIERARCHICAL,
        'hierarchical_using_expansion' : deesse.PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION,
        'all_level_one_by_one'         : deesse.PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE,
        'pyramid_sim_none'             : deesse.PYRAMID_SIM_NONE
    }
    try:
        pgp_c.pyramidSimulationMode = pyramidSimulationMode_dict[pgp.pyramidSimulationMode]
    except:
        pgp_c.pyramidSimulationMode = pyramidSimulationMode_dict['pyramid_sim_none']

    if nl > 0:
        # ... kx
        pgp_c.kx = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            pgp_c.kx, 0,
                np.asarray(pgp.kx, dtype='intc').reshape(nl))

        # ... ky
        pgp_c.ky = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            pgp_c.ky, 0,
                np.asarray(pgp.ky, dtype='intc').reshape(nl))

        # ... kz
        pgp_c.kz = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
            pgp_c.kz, 0,
                np.asarray(pgp.kz, dtype='intc').reshape(nl))

        # ... factorNneighboringNode and factorDistanceThreshold ...
        if pgp.pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            nn = 4*nl + 1
        else: # pyramidSimulationMode == 'all_level_one_by_one'
            nn = nl + 1

        # ... factorNneighboringNode
        pgp_c.factorNneighboringNode = deesse.new_double_array(nn)
        deesse.mpds_set_double_vector_from_array(
            pgp_c.factorNneighboringNode, 0,
                np.asarray(pgp.factorNneighboringNode).reshape(nn))

        # ... factorDistanceThreshold
        pgp_c.factorDistanceThreshold = deesse.new_real_array(nn)
        deesse.mpds_set_real_vector_from_array(
            pgp_c.factorDistanceThreshold, 0,
                np.asarray(pgp.factorDistanceThreshold).reshape(nn))

        # ... factorMaxScanFraction
        pgp_c.factorMaxScanFraction = deesse.new_double_array(nl+1)
        deesse.mpds_set_double_vector_from_array(
            pgp_c.factorMaxScanFraction, 0,
                np.asarray(pgp.factorMaxScanFraction).reshape(nl+1))

    return pgp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidGeneralParameters_C2py(pgp_c):
    """
    Converts pyramid general parameters from C to python.

    :param pgp_c:   (MPDS_PYRAMIDGENERALPARAMETERS *) corresponding parameters
                        (C struct)
    :return pgp:    (PyramidGeneralParameters class) pyramid general parameters
                        (python)
    """

    # ... npyramidLevel
    npyramidLevel = pgp_c.npyramidLevel
    if npyramidLevel == 0:
        pgp = PyramidGeneralParameters()
        return pgp

    # default parameters
    kx=None
    ky=None
    kz=None
    pyramidSimulationMode = 'hierarchical_using_expansion'
    factorNneighboringNode=None
    factorDistanceThreshold=None
    factorMaxScanFraction=None

    # ... pyramidSimulationMode
    pyramidSimulationMode_dict = {
        deesse.PYRAMID_SIM_HIERARCHICAL                :  'hierarchical',
        deesse.PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION : 'hierarchical_using_expansion',
        deesse.PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE         : 'all_level_one_by_one',
        deesse.PYRAMID_SIM_NONE                         : 'pyramid_sim_none'
    }
    try:
        pyramidSimulationMode = pyramidSimulationMode_dict[pgp_c.pyramidSimulationMode]
    except:
        pyramidSimulationMode = pyramidSimulationMode_dict['pyramid_sim_none']

    if npyramidLevel > 0:
        # ... kx
        kx = np.zeros(npyramidLevel, dtype='intc')
        deesse.mpds_get_array_from_int_vector(pgp_c.kx, 0, kx)
        kx = kx.astype('int')

        # ... ky
        ky = np.zeros(npyramidLevel, dtype='intc')
        deesse.mpds_get_array_from_int_vector(pgp_c.ky, 0, ky)
        ky = ky.astype('int')

        # ... kz
        kz = np.zeros(npyramidLevel, dtype='intc')
        deesse.mpds_get_array_from_int_vector(pgp_c.kz, 0, kz)
        kz = kz.astype('int')


        # ... factorNneighboringNode and factorDistanceThreshold ...
        if pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            nn = 4*npyramidLevel + 1
        else: # pyramidSimulationMode == 'all_level_one_by_one'
            nn = npyramidLevel + 1

        # ... factorNneighboringNode
        factorNneighboringNode = np.zeros(nn, dtype='double')
        deesse.mpds_get_array_from_double_vector(pgp_c.factorNneighboringNode, 0, factorNneighboringNode)
        factorNneighboringNode = factorNneighboringNode.astype('float')

        # ... factorDistanceThreshold
        factorDistanceThreshold = np.zeros(nn, dtype='float')
        deesse.mpds_get_array_from_real_vector(pgp_c.factorDistanceThreshold, 0, factorDistanceThreshold)

        # ... factorMaxScanFraction
        factorMaxScanFraction = np.zeros(npyramidLevel+1, dtype='double')
        deesse.mpds_get_array_from_double_vector(pgp_c.factorMaxScanFraction, 0, factorMaxScanFraction)
        factorMaxScanFraction = factorMaxScanFraction.astype('float')

    pgp = PyramidGeneralParameters(
        npyramidLevel=npyramidLevel,
        kx=kx, ky=ky, kz=kz,
        pyramidSimulationMode=pyramidSimulationMode,
        factorNneighboringNode=factorNneighboringNode,
        factorDistanceThreshold=factorDistanceThreshold,
        factorMaxScanFraction=factorMaxScanFraction
    )

    return pgp
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidParameters_py2C(pp):
    """
    Converts pyramid parameters from python to C.

    :param pp:      (PyramidParameters class) pyramid parameters (python)
    :return pp_c:   (MPDS_PYRAMIDPARAMETERS *) corresponding parameters (C struct)
    """

    fname = 'pyramidParameters_py2C'

    pp_c = deesse.malloc_MPDS_PYRAMIDPARAMETERS()
    deesse.MPDSInitPyramidParameters(pp_c)

    # ... nlevel
    pp_c.nlevel = int(pp.nlevel)

    # ... pyramidType
    pyramidType_dict = {
        'none'                      : deesse.PYRAMID_NONE,
        'continuous'                : deesse.PYRAMID_CONTINUOUS,
        'categorical_auto'          : deesse.PYRAMID_CATEGORICAL_AUTO,
        'categorical_custom'        : deesse.PYRAMID_CATEGORICAL_CUSTOM,
        'categorical_to_continuous' : deesse.PYRAMID_CATEGORICAL_TO_CONTINUOUS
    }
    try:
        pp_c.pyramidType = pyramidType_dict[pp.pyramidType]
    except:
        print(f'ERROR ({fname}): pyramid type unknown')
        return None

    if pp.pyramidType == 'categorical_custom':
        # ... classOfValues
        pp_c.classOfValues = classInterval2classOfValues(pp.classInterval)

    # ... outputLevelFlag
    deesse.mpds_allocate_and_set_pyramid_outputLevelFlag(pp_c, np.array([int(i) for i in pp.outputLevelFlag], dtype='intc'))

    return pp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidParameters_C2py(pp_c):
    """
    Converts pyramid parameters from C to python.

    :param pp_c:    (MPDS_PYRAMIDPARAMETERS *) corresponding parameters (C struct)
    :return pp:     (PyramidParameters class) pyramid parameters (python)
    """

    fname = 'pyramidParameters_C2py'

    # ... nlevel
    nlevel = pp_c.nlevel
    if nlevel == 0:
        pp = PyramidParameters()
        return pp

    # default parameters
    pyramidType='none'
    nclass=0
    classInterval=None
    outputLevelFlag=None

    # ... pyramidType
    pyramidType_dict = {
        deesse.PYRAMID_NONE                      : 'none',
        deesse.PYRAMID_CONTINUOUS                : 'continuous',
        deesse.PYRAMID_CATEGORICAL_AUTO          : 'categorical_auto',
        deesse.PYRAMID_CATEGORICAL_CUSTOM        : 'categorical_custom',
        deesse.PYRAMID_CATEGORICAL_TO_CONTINUOUS : 'categorical_to_continuous'
    }
    try:
        pyramidType = pyramidType_dict[pp_c.pyramidType]
    except:
        print(f'ERROR ({fname}): pyramid type unknown')
        return None

    if pyramidType == 'categorical_custom':
        # ... classInterval
        classInterval = classOfValues2classInterval(pp_c.classOfValues)
        nclass = len(classInterval)

    # ... outputLevelFlag
    outputLevelFlag = np.zeros(nlevel, dtype='intc')
    deesse.mpds_get_pyramid_outputLevelFlag(pp_c, outputLevelFlag)
    outputLevelFlag = outputLevelFlag.astype('bool')

    pp = PyramidParameters(
         nlevel=nlevel,
         pyramidType=pyramidType,
         nclass=nclass,
         classInterval=classInterval,
         outputLevelFlag=outputLevelFlag
    )

    return pp
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_input_py2C(deesse_input):
    """
    Converts deesse input from python to C.

    :param deesse_input:    (DeesseInput class) deesse input - python
    :return:                (MPDS_SIMINPUT *) deesse input - C
    """

    fname = 'deesse_input_py2C'

    nx = int(deesse_input.nx)
    ny = int(deesse_input.ny)
    nz = int(deesse_input.nz)
    sx = float(deesse_input.sx)
    sy = float(deesse_input.sy)
    sz = float(deesse_input.sz)
    ox = float(deesse_input.ox)
    oy = float(deesse_input.oy)
    oz = float(deesse_input.oz)
    nv = int(deesse_input.nv)

    nTI = int(deesse_input.nTI)

    nxy = nx * ny
    nxyz = nxy * nz

    # Allocate mpds_siminput
    mpds_siminput = deesse.malloc_MPDS_SIMINPUT()

    # Init mpds_siminput
    deesse.MPDSInitSimInput(mpds_siminput)

    # mpds_siminput.consoleAppFlag
    if deesse_input.consoleAppFlag:
        mpds_siminput.consoleAppFlag = deesse.TRUE
    else:
        mpds_siminput.consoleAppFlag = deesse.FALSE

    # mpds_siminput.simName
    # (mpds_siminput.simName not used, but must be set (could be '')!
    if not isinstance(deesse_input.simName, str):
        print(f'ERROR ({fname}): simName is not a string')
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None
    if len(deesse_input.simName) >= deesse.MPDS_VARNAME_LENGTH:
        print(f'ERROR ({fname}): simName is too long')
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None

    deesse.mpds_allocate_and_set_simname(mpds_siminput, deesse_input.simName)
    # mpds_siminput.simName = deesse_input.simName #  works too

    # mpds_siminput.simImage ...
    # ... set initial image im (for simulation)
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=nv, val=deesse.MPDS_MISSING_VALUE,
             varname=deesse_input.varname)

    # ... convert im from python to C
    mpds_siminput.simImage = img_py2C(im)
    if mpds_siminput.simImage is None:
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None

    # mpds_siminput.nvar
    mpds_siminput.nvar = nv

    # mpds_siminput.outputVarFlag
    deesse.mpds_allocate_and_set_outputVarFlag(mpds_siminput, deesse_input.outputVarFlag)
    # deesse.mpds_allocate_and_set_outputVarFlag(mpds_siminput, np.array([int(i) for i in deesse_input.outputVarFlag], dtype='bool'))

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
        deesse.mpds_allocate_and_set_outputReportFileName(mpds_siminput, deesse_input.outputReportFileName)
    else:
        mpds_siminput.outputReportFlag = deesse.FALSE

    # mpds_siminput.ntrainImage
    mpds_siminput.ntrainImage = nTI

    # mpds_siminput.simGridAsTiFlag
    deesse.mpds_allocate_and_set_simGridAsTiFlag(mpds_siminput, deesse_input.simGridAsTiFlag)
    # deesse.mpds_allocate_and_set_simGridAsTiFlag(mpds_siminput, np.array([int(i) for i in deesse_input.simGridAsTiFlag], dtype='bool')) # dtype='intc'))

    # mpds_siminput.trainImage
    mpds_siminput.trainImage = deesse.new_MPDS_IMAGE_array(nTI)
    for i, ti in enumerate(deesse_input.TI):
        if ti is not None:
            im_c = img_py2C(ti)
            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.trainImage, i, im_c)
            # deesse.free_MPDS_IMAGE(im_c)
            #
            # deesse.MPDS_IMAGE_array_setitem(mpds_siminput.trainImage, i, img_py2C(ti))

    # mpds_siminput.pdfTrainImage
    if nTI > 1:
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=nTI, val=deesse_input.pdfTI)
        mpds_siminput.pdfTrainImage = img_py2C(im)

    # mpds_siminput.ndataImage and mpds_siminput.dataImage
    if deesse_input.dataImage is None:
        mpds_siminput.ndataImage = 0
    else:
        n = len(deesse_input.dataImage)
        mpds_siminput.ndataImage = n
        mpds_siminput.dataImage = deesse.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(deesse_input.dataImage):
            im_c = img_py2C(dataIm)
            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImage, i, im_c)
            # deesse.free_MPDS_IMAGE(im_c)
            #
            # deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImage, i, img_py2C(dataIm))

    # mpds_siminput.ndataPointSet and mpds_siminput.dataPointSet
    if deesse_input.dataPointSet is None:
        mpds_siminput.ndataPointSet = 0
    else:
        n = len(deesse_input.dataPointSet)
        mpds_siminput.ndataPointSet = n
        mpds_siminput.dataPointSet = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesse_input.dataPointSet):
            ps_c = ps_py2C(dataPS)
            deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSet, i, ps_c)
            # deesse.free_MPDS_POINTSET(ps_c)
            #
            # deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSet, i, ps_py2C(dataPS))

    # mpds_siminput.maskImageFlag and mpds_siminput.maskImage
    if deesse_input.mask is None:
        mpds_siminput.maskImageFlag = deesse.FALSE
    else:
        mpds_siminput.maskImageFlag = deesse.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
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
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=deesse_input.homothetyXRatio)
            mpds_siminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyXRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyXRatioValue, 0,
                np.asarray(deesse_input.homothetyXRatio).reshape(1))

        if deesse_input.homothetyYLocal:
            mpds_siminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=deesse_input.homothetyYRatio)
            mpds_siminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyYRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyYRatioValue, 0,
                np.asarray(deesse_input.homothetyYRatio).reshape(1))

        if deesse_input.homothetyZLocal:
            mpds_siminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
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
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=deesse_input.homothetyXRatio)
            mpds_siminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyXRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyXRatioValue, 0,
                np.asarray(deesse_input.homothetyXRatio).reshape(2))

        if deesse_input.homothetyYLocal:
            mpds_siminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=deesse_input.homothetyYRatio)
            mpds_siminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_siminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_siminput.homothetyYRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.homothetyYRatioValue, 0,
                np.asarray(deesse_input.homothetyYRatio).reshape(2))

        if deesse_input.homothetyZLocal:
            mpds_siminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
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
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=deesse_input.rotationAzimuth)
            mpds_siminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_siminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_siminput.rotationAzimuthValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationAzimuthValue, 0,
                np.asarray(deesse_input.rotationAzimuth).reshape(1))

        if deesse_input.rotationDipLocal:
            mpds_siminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=deesse_input.rotationDip)
            mpds_siminput.rotationDipImage = img_py2C(im)

        else:
            mpds_siminput.rotationDipImageFlag = deesse.FALSE
            mpds_siminput.rotationDipValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationDipValue, 0,
                np.asarray(deesse_input.rotationDip).reshape(1))

        if deesse_input.rotationPlungeLocal:
            mpds_siminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
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
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=deesse_input.rotationAzimuth)
            mpds_siminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_siminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_siminput.rotationAzimuthValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationAzimuthValue, 0,
                np.asarray(deesse_input.rotationAzimuth).reshape(2))

        if deesse_input.rotationDipLocal:
            mpds_siminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=deesse_input.rotationDip)
            mpds_siminput.rotationDipImage = img_py2C(im)

        else:
            mpds_siminput.rotationDipImageFlag = deesse.FALSE
            mpds_siminput.rotationDipValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_siminput.rotationDipValue, 0,
                np.asarray(deesse_input.rotationDip).reshape(2))

        if deesse_input.rotationPlungeLocal:
            mpds_siminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
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
    normalizingType_dict = {
        'linear'  : deesse.NORMALIZING_LINEAR,
        'uniform' : deesse.NORMALIZING_UNIFORM,
        'normal'  : deesse.NORMALIZING_NORMAL
    }
    try:
        mpds_siminput.normalizingType = normalizingType_dict[deesse_input.normalizingType]
    except:
        print(f'ERROR ({fname}): normalizing type unknown')
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None

    # mpds_siminput.searchNeighborhoodParameters
    mpds_siminput.searchNeighborhoodParameters = deesse.new_MPDS_SEARCHNEIGHBORHOODPARAMETERS_array(nv)
    for i, sn in enumerate(deesse_input.searchNeighborhoodParameters):
        sn_c = search_neighborhood_parameters_py2C(sn)
        if sn_c is None:
            print(f'ERROR ({fname}): can not convert search neighborhood parameters from python to C')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None
        deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_setitem(
            mpds_siminput.searchNeighborhoodParameters, i, sn_c)
        # deesse.free_MPDS_SEARCHNEIGHBORHOODPARAMETERS(sn_c)

    # mpds_siminput.nneighboringNode
    mpds_siminput.nneighboringNode = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.nneighboringNode, 0,
        np.asarray(deesse_input.nneighboringNode, dtype='intc').reshape(nv))

    # mpds_siminput.maxPropInequalityNode
    mpds_siminput.maxPropInequalityNode = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.maxPropInequalityNode, 0,
        np.asarray(deesse_input.maxPropInequalityNode).reshape(nv))

    # mpds_siminput.neighboringNodeDensity
    mpds_siminput.neighboringNodeDensity = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.neighboringNodeDensity, 0,
        np.asarray(deesse_input.neighboringNodeDensity).reshape(nv))

    # mpds_siminput.rescalingMode
    rescalingMode_dict = {
        'none'        : deesse.RESCALING_NONE,
        'min_max'     : deesse.RESCALING_MIN_MAX,
        'mean_length' : deesse.RESCALING_MEAN_LENGTH
    }
    mpds_siminput.rescalingMode = deesse.new_MPDS_RESCALINGMODE_array(nv)
    for i, m in enumerate(deesse_input.rescalingMode):
        if m in rescalingMode_dict.keys():
            deesse.MPDS_RESCALINGMODE_array_setitem(mpds_siminput.rescalingMode, i, rescalingMode_dict[m])
        else:
            print(f'ERROR ({fname}): rescaling mode unknown')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None

    # mpds_simInput.rescalingTargetMin
    mpds_siminput.rescalingTargetMin = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMin, 0,
        np.asarray(deesse_input.rescalingTargetMin).reshape(nv))

    # mpds_simInput.rescalingTargetMax
    mpds_siminput.rescalingTargetMax = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMax, 0,
        np.asarray(deesse_input.rescalingTargetMax).reshape(nv))

    # mpds_simInput.rescalingTargetMean
    mpds_siminput.rescalingTargetMean = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetMean, 0,
        np.asarray(deesse_input.rescalingTargetMean).reshape(nv))

    # mpds_simInput.rescalingTargetLength
    mpds_siminput.rescalingTargetLength = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.rescalingTargetLength, 0,
        np.asarray(deesse_input.rescalingTargetLength).reshape(nv))

    # mpds_siminput.relativeDistanceFlag
    deesse.mpds_allocate_and_set_relativeDistanceFlag(mpds_siminput, deesse_input.relativeDistanceFlag)
    # deesse.mpds_allocate_and_set_relativeDistanceFlag(mpds_siminput, np.array([int(i) for i in deesse_input.relativeDistanceFlag], dtype='bool')) # , dtype='intc'))

    # mpds_siminput.distanceType
    mpds_siminput.distanceType = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.distanceType, 0,
        np.asarray(deesse_input.distanceType, dtype='intc').reshape(nv))

    # mpds_siminput.powerLpDistance
    mpds_siminput.powerLpDistance = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.powerLpDistance, 0,
        np.asarray(deesse_input.powerLpDistance).reshape(nv))

    # mpds_siminput.powerLpDistanceInv
    mpds_siminput.powerLpDistanceInv = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.powerLpDistanceInv, 0,
        np.asarray(deesse_input.powerLpDistanceInv).reshape(nv))

    # mpds_siminput.conditioningWeightFactor
    mpds_siminput.conditioningWeightFactor = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.conditioningWeightFactor, 0,
        np.asarray(deesse_input.conditioningWeightFactor).reshape(nv))

    # mpds_siminput.simAndPathParameters
    mpds_siminput.simAndPathParameters = set_simAndPathParameters_C(
        deesse_input.simType,
        deesse_input.simPathType,
        deesse_input.simPathStrength,
        deesse_input.simPathPower,
        deesse_input.simPathUnilateralOrder)
    if mpds_siminput.simAndPathParameters is None:
        print(f'ERROR ({fname}): can not set "simAndPathParameters" in C')
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None

    # mpds_siminput.distanceThreshold
    mpds_siminput.distanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.distanceThreshold, 0,
        np.asarray(deesse_input.distanceThreshold).reshape(nv))

    # mpds_siminput.softProbability ...
    mpds_siminput.softProbability = deesse.new_MPDS_SOFTPROBABILITY_array(nv)

    # ... for each variable ...
    for i, sp in enumerate(deesse_input.softProbability):
        sp_c = softProbability_py2C(sp,
                                    nx, ny, nz,
                                    sx, sy, sz,
                                    ox, oy, oz)
        if sp_c is None:
            print(f'ERROR ({fname}): can not set soft probability parameters in C')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None
        deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_siminput.softProbability, i, sp_c)
        # deesse.free_MPDS_SOFTPROBABILITY(sp_c)

    # mpds_siminput.connectivity ...
    mpds_siminput.connectivity = deesse.new_MPDS_CONNECTIVITY_array(nv)

    for i, co in enumerate(deesse_input.connectivity):
        co_c = connectivity_py2C(co)
        if co_c is None:
            print(f'ERROR ({fname}): can not set connectivity parameters in C')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None
        deesse.MPDS_CONNECTIVITY_array_setitem(mpds_siminput.connectivity, i, co_c)
        # deesse.free_MPDS_CONNECTIVITY(co_c)

    # mpds_siminput.blockData ...
    mpds_siminput.blockData = deesse.new_MPDS_BLOCKDATA_array(nv)
    # ... for each variable ...
    for i, bd in enumerate(deesse_input.blockData):
        bd_c = blockData_py2C(bd)
        if bd_c is None:
            print(f'ERROR ({fname}): can not set block data parameters in C')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None
        deesse.MPDS_BLOCKDATA_array_setitem(mpds_siminput.blockData, i, bd_c)
        # deesse.free_MPDS_BLOCKDATA(bd_c)

    # mpds_siminput.maxScanFraction
    mpds_siminput.maxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.maxScanFraction, 0,
            np.asarray(deesse_input.maxScanFraction).reshape(nTI))

    # mpds_siminput.pyramidGeneralParameters ...
    mpds_siminput.pyramidGeneralParameters = pyramidGeneralParameters_py2C(deesse_input.pyramidGeneralParameters)
    if mpds_siminput.pyramidGeneralParameters is None:
        print(f'ERROR ({fname}): can not set pyramid general parameters in C')
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        #deesse.MPDSFree(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        return None

    # mpds_siminput.pyramidParameters ...
    mpds_siminput.pyramidParameters = deesse.new_MPDS_PYRAMIDPARAMETERS_array(nv)

    # ... for each variable ...
    for i, pp in enumerate(deesse_input.pyramidParameters):
        pp_c = pyramidParameters_py2C(pp)
        if pp_c is None:
            print(f'ERROR ({fname}): can not set pyramid parameters in C')
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            #deesse.MPDSFree(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            return None

        deesse.MPDS_PYRAMIDPARAMETERS_array_setitem(mpds_siminput.pyramidParameters, i, pp_c)
        # deesse.free_MPDS_PYRAMIDPARAMETERS(pp_c)

    # mpds_siminput.ndataImageInPyramid and mpds_siminput.dataImageInPyramid
    if deesse_input.pyramidDataImage is None:
        mpds_siminput.ndataImageInPyramid = 0
    else:
        n = len(deesse_input.pyramidDataImage)
        mpds_siminput.ndataImageInPyramid = n
        mpds_siminput.dataImageInPyramid = deesse.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(deesse_input.pyramidDataImage):
            im_c = img_py2C(dataIm)
            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImageInPyramid, i, im_c)
            # deesse.free_MPDS_IMAGE(im_c)
            #
            # deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImageInPyramid, i, img_py2C(dataIm))

    # mpds_siminput.ndataPointSetInPyramid and mpds_siminput.dataPointSetInPyramid
    if deesse_input.pyramidDataPointSet is None:
        mpds_siminput.ndataPointSetInPyramid = 0
    else:
        n = len(deesse_input.pyramidDataPointSet)
        mpds_siminput.ndataPointSetInPyramid = n
        mpds_siminput.dataPointSetInPyramid = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesse_input.pyramidDataPointSet):
            ps_c = ps_py2C(dataPS)
            deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSetInPyramid, i, ps_c)
            # deesse.free_MPDSPOINTSET(ps_c)
            #
            # deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSetInPyramid, i, ps_py2C(dataPS))

    # mpds_siminput.tolerance
    mpds_siminput.tolerance = deesse_input.tolerance

    # mpds_siminput.npostProcessingPathMax
    mpds_siminput.npostProcessingPathMax = deesse_input.npostProcessingPathMax

    # mpds_siminput.postProcessingNneighboringNode
    mpds_siminput.postProcessingNneighboringNode = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_siminput.postProcessingNneighboringNode, 0,
            np.asarray(deesse_input.postProcessingNneighboringNode, dtype='intc').reshape(nv))

    # mpds_siminput.postProcessingNeighboringNodeDensity
    mpds_siminput.postProcessingNeighboringNodeDensity = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.postProcessingNeighboringNodeDensity, 0,
            np.asarray(deesse_input.postProcessingNeighboringNodeDensity).reshape(nv))

    # mpds_siminput.postProcessingDistanceThreshold
    mpds_siminput.postProcessingDistanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.postProcessingDistanceThreshold, 0,
            np.asarray(deesse_input.postProcessingDistanceThreshold).reshape(nv))

    # mpds_siminput.postProcessingMaxScanFraction
    mpds_siminput.postProcessingMaxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.postProcessingMaxScanFraction, 0,
            np.asarray(deesse_input.postProcessingMaxScanFraction).reshape(nTI))

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
def deesse_input_C2py(mpds_siminput):
    """
    Converts deesse input from C to python.

    :param mpds_siminput:   (MPDS_SIMINPUT *) deesse input - C
    :return:                (DeesseInput class) deesse input - python
    """

    fname = 'deesse_input_C2py'

    # simName
    simName = mpds_siminput.simName

    im = img_C2py(mpds_siminput.simImage)

    # grid geometry
    nx = im.nx
    ny = im.ny
    nz = im.nz
    sx = im.sx
    sy = im.sy
    sz = im.sz
    ox = im.ox
    oy = im.oy
    oz = im.oz

    # nv (number of variable(s))
    nv = im.nv # or: nv = int(mpds_siminput.nvar)

    # varname
    varname = im.varname

    # outputVarFlag
    outputVarFlag = np.zeros(nv, dtype='intc')
    deesse.mpds_get_outputVarFlag(mpds_siminput, outputVarFlag)
    outputVarFlag = outputVarFlag.astype('bool')

    # output maps
    outputPathIndexFlag       = bool(int.from_bytes(mpds_siminput.outputPathIndexFlag.encode('utf-8'), byteorder='big'))
    outputErrorFlag           = bool(int.from_bytes(mpds_siminput.outputErrorFlag.encode('utf-8'), byteorder='big'))
    outputTiGridNodeIndexFlag = bool(int.from_bytes(mpds_siminput.outputTiGridNodeIndexFlag.encode('utf-8'), byteorder='big'))
    outputTiIndexFlag         = bool(int.from_bytes(mpds_siminput.outputTiIndexFlag.encode('utf-8'), byteorder='big'))
    outputReportFlag          = bool(int.from_bytes(mpds_siminput.outputReportFlag.encode('utf-8'), byteorder='big'))

    # report
    if outputReportFlag:
        outputReportFileName = mpds_siminput.outputReportFileName
    else:
        outputReportFileName = None

    # TI, simGridAsTiFlag, nTI
    nTI = mpds_siminput.ntrainImage
    simGridAsTiFlag = np.zeros(nTI, dtype='intc')
    deesse.mpds_get_simGridAsTiFlag(mpds_siminput, simGridAsTiFlag)
    simGridAsTiFlag = simGridAsTiFlag.astype('bool')
    TI = np.array(nTI*[None])
    for i in range(nTI):
        if not simGridAsTiFlag[i]:
            im = deesse.MPDS_IMAGE_array_getitem(mpds_siminput.trainImage, i)
            TI[i] = img_C2py(im)

    # pdfTI
    pdfTI = None
    if nTI > 1:
        im = img_C2py(mpds_siminput.pdfTrainImage)
        pdfTI = im.val

    # dataImage
    dataImage = None
    ndataImage = mpds_siminput.ndataImage
    if ndataImage > 0:
        dataImage = np.array(ndataImage*[None])
        for i in range(ndataImage):
            im = deesse.MPDS_IMAGE_array_getitem(mpds_siminput.dataImage, i)
            dataImage[i] = img_C2py(im)

    # dataPointSet
    dataPointSet = None
    ndataPointSet = mpds_siminput.ndataPointSet
    if ndataPointSet > 0:
        dataPointSet = np.array(ndataPointSet*[None])
        for i in range(ndataPointSet):
            ps = deesse.MPDS_POINTSET_array_getitem(mpds_siminput.dataPointSet, i)
            dataPointSet[i] = ps_C2py(ps)

    # mask
    mask = None
    flag = bool(int.from_bytes(mpds_siminput.maskImageFlag.encode('utf-8'), byteorder='big'))
    if flag:
        im = img_C2py(mpds_siminput.maskImage)
        mask = im.val

    # homothety
    homothetyUsage = mpds_siminput.homothetyUsage
    homothetyXLocal = False
    homothetyXRatio = None
    homothetyYLocal = False
    homothetyYRatio = None
    homothetyZLocal = False
    homothetyZRatio = None
    if homothetyUsage == 1:
        homothetyXLocal = bool(int.from_bytes(mpds_siminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_siminput.homothetyXRatioImage)
            homothetyXRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyXRatioValue, 0, v)
            homothetyXRatio = v[0]

        homothetyYLocal = bool(int.from_bytes(mpds_siminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_siminput.homothetyYRatioImage)
            homothetyYRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyYRatioValue, 0, v)
            homothetyYRatio = v[0]

        homothetyZLocal = bool(int.from_bytes(mpds_siminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_siminput.homothetyZRatioImage)
            homothetyZRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyZRatioValue, 0, v)
            homothetyZRatio = v[0]

    elif homothetyUsage == 2:
        homothetyXLocal = bool(int.from_bytes(mpds_siminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_siminput.homothetyXRatioImage)
            homothetyXRatio = im.val
        else:
            homothetyXRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyXRatioValue, 0, homothetyXRatio)

        homothetyYLocal = bool(int.from_bytes(mpds_siminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_siminput.homothetyYRatioImage)
            homothetyYRatio = im.val
        else:
            homothetyYRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyYRatioValue, 0, homothetyYRatio)

        homothetyZLocal = bool(int.from_bytes(mpds_siminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_siminput.homothetyZRatioImage)
            homothetyZRatio = im.val
        else:
            homothetyZRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyZRatioValue, 0, homothetyZRatio)

    # rotation
    rotationUsage = mpds_siminput.rotationUsage
    rotationAzimuthLocal = False
    rotationAzimuth = None
    rotationDipLocal = False
    rotationDip = None
    rotationPlungeLocal = False
    rotationPlunge = None
    if rotationUsage == 1:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_siminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_siminput.rotationAzimuthImage)
            rotationAzimuth = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationAzimuthValue, 0, v)
            rotationAzimuth = v[0]

        rotationDipLocal = bool(int.from_bytes(mpds_siminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_siminput.rotationDipImage)
            rotationDip = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationDipValue, 0, v)
            rotationDip = v[0]

        rotationPlungeLocal = bool(int.from_bytes(mpds_siminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_siminput.rotationPlungeImage)
            rotationPlunge = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationPlungeValue, 0, v)
            rotationPlunge = v[0]

    elif rotationUsage == 2:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_siminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_siminput.rotationAzimuthImage)
            rotationAzimuth = im.val
        else:
            rotationAzimuth = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationAzimuthValue, 0, rotationAzimuth)

        rotationDipLocal = bool(int.from_bytes(mpds_siminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_siminput.rotationDipImage)
            rotationDip = im.val
        else:
            rotationDip = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationDipValue, 0, rotationDip)

        rotationPlungeLocal = bool(int.from_bytes(mpds_siminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_siminput.rotationPlungeImage)
            rotationPlunge = im.val
        else:
            rotationPlunge = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationPlungeValue, 0, rotationPlunge)

    # expMax
    expMax = mpds_siminput.trainValueRangeExtensionMax

    # normalizingType
    normalizingType_dict = {
        deesse.NORMALIZING_LINEAR  : 'linear',
        deesse.NORMALIZING_UNIFORM : 'uniform',
        deesse.NORMALIZING_NORMAL  : 'normal'
    }
    try:
        normalizingType = normalizingType_dict[mpds_siminput.normalizingType]
    except:
        print(f'ERROR ({fname}): normalizing type unknown')
        return None

    # searchNeighborhoodParameters
    searchNeighborhoodParameters = np.array(nv*[None])
    for i in range(nv):
        sn_c = deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_getitem(mpds_siminput.searchNeighborhoodParameters, i)
        sn = search_neighborhood_parameters_C2py(sn_c)
        if sn is None:
            print(f'ERROR ({fname}): can not convert search neighborhood parameters from C to python')
            return None
        searchNeighborhoodParameters[i] = sn

    # nneighboringNode
    nneighboringNode = np.zeros(nv, dtype='intc')
    deesse.mpds_get_array_from_int_vector(mpds_siminput.nneighboringNode, 0, nneighboringNode)
    nneighboringNode = nneighboringNode.astype('int')

    # maxPropInequalityNode
    maxPropInequalityNode = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_siminput.maxPropInequalityNode, 0, maxPropInequalityNode)
    maxPropInequalityNode = maxPropInequalityNode.astype('float')

    # neighboringNodeDensity
    neighboringNodeDensity = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_siminput.neighboringNodeDensity, 0, neighboringNodeDensity)
    neighboringNodeDensity = neighboringNodeDensity.astype('float')

    # rescalingMode
    rescalingMode_dict = {
        deesse.RESCALING_NONE        : 'none',
        deesse.RESCALING_MIN_MAX     : 'min_max',
        deesse.RESCALING_MEAN_LENGTH : 'mean_length'
    }
    rescalingMode = np.array(nv*[None])
    for i in range(nv):
        rs_c = deesse.MPDS_RESCALINGMODE_array_getitem(mpds_siminput.rescalingMode, i)
        try:
            rs = rescalingMode_dict[rs_c]
            rescalingMode[i] = rs
        except:
            print(f'ERROR ({fname}): rescaling mode unknown')
            return None

    # rescalingTargetMin
    rescalingTargetMin = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.rescalingTargetMin, 0, rescalingTargetMin)

    # rescalingTargetMax
    rescalingTargetMax = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.rescalingTargetMax, 0, rescalingTargetMax)

    # rescalingTargetMean
    rescalingTargetMean = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.rescalingTargetMean, 0, rescalingTargetMean)

    # rescalingTargetLength
    rescalingTargetLength = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.rescalingTargetLength, 0, rescalingTargetLength)

    # relativeDistanceFlag
    relativeDistanceFlag = np.zeros(nv, dtype='intc')
    deesse.mpds_get_relativeDistanceFlag(mpds_siminput, relativeDistanceFlag)
    relativeDistanceFlag = relativeDistanceFlag.astype('bool')

    # distanceType
    distanceType = np.zeros(nv, dtype='intc')
    deesse.mpds_get_array_from_int_vector(mpds_siminput.distanceType, 0, distanceType)
    distanceType = distanceType.astype('int')
    distanceType = list(distanceType)
    for i in range(nv):
        if distanceType[i] == 0:
            distanceType[i] = 'categorical'
        elif distanceType[i] == 1:
            distanceType[i] = 'continuous'

    # powerLpDistance
    powerLpDistance = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_siminput.powerLpDistance, 0, powerLpDistance)
    powerLpDistance = powerLpDistance.astype('float')
    for i in range(nv):
        if distanceType[i] != 3:
            powerLpDistance[i] = 1.0

    # conditioningWeightFactor
    conditioningWeightFactor = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.conditioningWeightFactor, 0, conditioningWeightFactor)

    # simType
    simType_c = mpds_siminput.simAndPathParameters.simType
    if simType_c == deesse.SIM_ONE_BY_ONE:
        simType = 'sim_one_by_one'
    elif simType_c == deesse.SIM_VARIABLE_VECTOR:
        simType = 'sim_variable_vector'
    else:
        print(f'ERROR ({fname}): simulation type unknown')
        return None

    # simPathType
    simPathType = None
    simPathPower = None
    simPathStrength = None
    simPathUnilateralOrder = None

    simPathType_c = mpds_siminput.simAndPathParameters.pathType
    if simPathType_c == deesse.PATH_RANDOM:
        simPathType = 'random'
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_PDF:
        simPathType = 'random_hd_distance_pdf'
        simPathStrength = mpds_siminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SORT:
        simPathType = 'random_hd_distance_sort'
        simPathStrength = mpds_siminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SUM_PDF:
        simPathType = 'random_hd_distance_sum_pdf'
        simPathPower = mpds_siminput.simAndPathParameters.pow
        simPathStrength = mpds_siminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SUM_SORT:
        simPathType = 'random_hd_distance_sum_sort'
        simPathPower = mpds_siminput.simAndPathParameters.pow
        simPathStrength = mpds_siminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_UNILATERAL:
        simPathType = 'unilateral'
        simPathUnilateralOrder = np.zeros(mpds_siminput.simAndPathParameters.unilateralOrderLength, dtype='intc')
        deesse.mpds_get_array_from_int_vector(mpds_siminput.simAndPathParameters.unilateralOrder, 0, simPathUnilateralOrder)
        simPathUnilateralOrder = simPathUnilateralOrder.astype('int')
    else:
        print(f'ERROR ({fname}): simulation path type unknown')
        return None

    # distanceThreshold
    distanceThreshold = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.distanceThreshold, 0, distanceThreshold)

    # softProbability
    softProbability = np.array(nv*[None])
    for i in range(nv):
        sp_c = deesse.MPDS_SOFTPROBABILITY_array_getitem(mpds_siminput.softProbability, i)
        sp = softProbability_C2py(sp_c)
        if sp is None:
            print(f'ERROR ({fname}): can not convert soft probability from C to python')
            return None
        softProbability[i] = sp

    # connectivity
    connectivity = np.array(nv*[None])
    for i in range(nv):
        co_c = deesse.MPDS_CONNECTIVITY_array_getitem(mpds_siminput.connectivity, i)
        co = connectivity_C2py(co_c)
        if co is None:
            print(f'ERROR ({fname}): can not convert connectivity parameters from C to python')
            return None
        connectivity[i] = co

    # blockData
    blockData = np.array(nv*[None])
    for i in range(nv):
        bd_c = deesse.MPDS_BLOCKDATA_array_getitem(mpds_siminput.blockData, i)
        bd = blockData_C2py(bd_c)
        if bd is None:
            print(f'ERROR ({fname}): can not convert block data parameters from C to python')
            return None
        blockData[i] = bd

    # maxScanFraction
    maxScanFraction = np.zeros(nTI, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_siminput.maxScanFraction, 0, maxScanFraction)
    maxScanFraction = maxScanFraction.astype('float')

    # pyramidGeneralParameters
    pyramidGeneralParameters = pyramidGeneralParameters_C2py(mpds_siminput.pyramidGeneralParameters)
    if pyramidGeneralParameters is None:
        print(f'ERROR ({fname}): can not convert pyramid general parameters from C to python')
        return None

    # pyramidParameters
    pyramidParameters = np.array(nv*[None])
    for i in range(nv):
        pp_c = deesse.MPDS_PYRAMIDPARAMETERS_array_getitem(mpds_siminput.pyramidParameters, i)
        pp = pyramidParameters_C2py(pp_c)
        if pp is None:
            print(f'ERROR ({fname}): can not convert pyramid parameters from C to python')
            return None
        pyramidParameters[i] = pp

    # pyramidDataImage
    pyramidDataImage = None
    npyramidDataImage = mpds_siminput.ndataImageInPyramid
    if npyramidDataImage > 0:
        pyramidDataImage = np.array(npyramidDataImage*[None])
        for i in range(npyramidDataImage):
            im = deesse.MPDS_IMAGE_array_getitem(mpds_siminput.dataImageInPyramid, i)
            pyramidDataImage[i] = img_C2py(im)

    # pyramidataPointSet
    pyramidDataPointSet = None
    npyramidDataPointSet = mpds_siminput.ndataPointSetInPyramid
    if npyramidDataPointSet > 0:
        pyramidDataPointSet = np.array(npyramidDataPointSet*[None])
        for i in range(npyramidDataPointSet):
            ps = deesse.MPDS_POINTSET_array_getitem(mpds_siminput.dataPointSetInPyramid, i)
            pyramidDataPointSet[i] = ps_C2py(ps)

    # tolerance
    tolerance = mpds_siminput.tolerance

    # npostProcessingPathMax
    npostProcessingPathMax = mpds_siminput.npostProcessingPathMax

    # default parameters
    postProcessingNneighboringNode = None
    postProcessingNeighboringNodeDensity = None
    postProcessingDistanceThreshold = None
    postProcessingMaxScanFraction = None
    postProcessingTolerance = 0.0

    if npostProcessingPathMax > 0:
        # postProcessingNneighboringNode
        postProcessingNneighboringNode = np.zeros(nv, dtype='intc')
        deesse.mpds_get_array_from_int_vector(mpds_siminput.postProcessingNneighboringNode, 0, postProcessingNneighboringNode)
        postProcessingNneighboringNode = postProcessingNneighboringNode.astype('int')

        # postProcessingNeighboringNodeDensity
        postProcessingNeighboringNodeDensity = np.zeros(nv, dtype='double')
        deesse.mpds_get_array_from_double_vector(mpds_siminput.postProcessingNeighboringNodeDensity, 0, postProcessingNeighboringNodeDensity)
        postProcessingNeighboringNodeDensity = postProcessingNeighboringNodeDensity.astype('float')

        # postProcessingDistanceThreshold
        postProcessingDistanceThreshold = np.zeros(nv, dtype='float')
        deesse.mpds_get_array_from_real_vector(mpds_siminput.postProcessingDistanceThreshold, 0, postProcessingDistanceThreshold)

        # mpds_siminput.postProcessingMaxScanFraction
        postProcessingMaxScanFraction = np.zeros(nTI, dtype='double')
        deesse.mpds_get_array_from_double_vector(mpds_siminput.postProcessingMaxScanFraction, 0, postProcessingMaxScanFraction)
        postProcessingMaxScanFraction = postProcessingMaxScanFraction.astype('float')

        # mpds_siminput.postProcessingTolerance
        postProcessingTolerance = mpds_siminput.postProcessingTolerance

    # seed
    seed = mpds_siminput.seed

    # seedIncrement
    seedIncrement = mpds_siminput.seedIncrement

    # nrealization
    nrealization = mpds_siminput.nrealization

    # deesse input
    deesse_input = DeesseInput(
        simName=simName,
        nx=nx, ny=ny, nz=nz,
        sx=sx, sy=sy, sz=sz,
        ox=ox, oy=oy, oz=oz,
        nv=nv, varname=varname, outputVarFlag=outputVarFlag,
        outputPathIndexFlag=outputPathIndexFlag,
        outputErrorFlag=outputErrorFlag,
        outputTiGridNodeIndexFlag=outputTiGridNodeIndexFlag,
        outputTiIndexFlag=outputTiIndexFlag,
        outputReportFlag=outputReportFlag, outputReportFileName=outputReportFileName,
        nTI=None, TI=TI, simGridAsTiFlag=simGridAsTiFlag, pdfTI=pdfTI,
        dataImage=dataImage, dataPointSet=dataPointSet,
        mask=mask,
        homothetyUsage=homothetyUsage,
        homothetyXLocal=homothetyXLocal, homothetyXRatio=homothetyXRatio,
        homothetyYLocal=homothetyYLocal, homothetyYRatio=homothetyYRatio,
        homothetyZLocal=homothetyZLocal, homothetyZRatio=homothetyZRatio,
        rotationUsage=rotationUsage,
        rotationAzimuthLocal=rotationAzimuthLocal, rotationAzimuth=rotationAzimuth,
        rotationDipLocal=rotationDipLocal,         rotationDip=rotationDip,
        rotationPlungeLocal=rotationPlungeLocal,   rotationPlunge=rotationPlunge,
        expMax=expMax,
        normalizingType=normalizingType,
        searchNeighborhoodParameters=searchNeighborhoodParameters,
        nneighboringNode=nneighboringNode,
        maxPropInequalityNode=maxPropInequalityNode, neighboringNodeDensity=neighboringNodeDensity,
        rescalingMode=rescalingMode,
        rescalingTargetMin=rescalingTargetMin, rescalingTargetMax=rescalingTargetMax,
        rescalingTargetMean=rescalingTargetMean, rescalingTargetLength=rescalingTargetLength,
        relativeDistanceFlag=relativeDistanceFlag,
        distanceType=distanceType,
        powerLpDistance=powerLpDistance,
        conditioningWeightFactor=conditioningWeightFactor,
        simType=simType,
        simPathType=simPathType,
        simPathStrength=simPathStrength,
        simPathPower=simPathPower,
        simPathUnilateralOrder=simPathUnilateralOrder,
        distanceThreshold=distanceThreshold,
        softProbability=softProbability,
        connectivity=connectivity,
        blockData=blockData,
        maxScanFraction=maxScanFraction,
        pyramidGeneralParameters=pyramidGeneralParameters,
        pyramidParameters=pyramidParameters,
        pyramidDataImage=pyramidDataImage, pyramidDataPointSet=pyramidDataPointSet,
        tolerance=tolerance,
        npostProcessingPathMax=npostProcessingPathMax,
        postProcessingNneighboringNode=postProcessingNneighboringNode,
        postProcessingNeighboringNodeDensity=postProcessingNeighboringNodeDensity,
        postProcessingDistanceThreshold=postProcessingDistanceThreshold,
        postProcessingMaxScanFraction=postProcessingMaxScanFraction,
        postProcessingTolerance=postProcessingTolerance,
        seed=seed,
        seedIncrement=seedIncrement,
        nrealization=nrealization)

    return deesse_input
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_output_C2py(mpds_simoutput, mpds_progressMonitor):
    """
    Get deesse output for python from C.

    :param mpds_simoutput:  (MPDS_SIMOUTPUT *) simulation output - (C struct)
                                contains output of deesse simulation
    :param mpds_progressMonitor:
                            (MPDS_PROGRESSMONITOR *) progress monitor - (C struct)
                                contains output messages (warnings) of deesse
                                simulation

    :return deesse_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'sim_pyramid':sim_pyramid,
             'sim_pyramid_var_original_index':sim_pyramid_var_original_index,
             'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
             'path':path,
             'error':error,
             'tiGridNode':tiGridNode,
             'tiIndex':tiIndex,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = mpds_simoutput->nreal (number of realizations):

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_simoutput->outputSimImage[0])
                    (sim is None if mpds_simoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_simoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_simoutput->originalVarIndex is NULL)

        sim_pyramid:
                (list or None) realizations in pyramid levels (depends on input
                parameters given in deesse_input); if pyramid was used and output
                in pyramid required:
                    sim_pyramid[j]:
                        (1-dimensional array of Img (class) of size nreal or None)
                        sim_pyramid[j][i]: i-th realisation in pyramid level of
                            index j+1, k-th variable stored refers to
                                - the original variable
                                    sim_pyramid_var_original_index[j][k]
                                - and pyramid index
                                    sim_pyramid_var_pyramid_index[j][k]
                            (get from
                            mpds_simoutput->outputSimImagePyramidLevel[j])
                        (sim_pyramid[j] is None if
                        mpds_simoutput->outputSimImagePyramidLevel[j] is NULL)
                (sim_pyramid is None otherwise)

        sim_pyramid_var_original_index:
                (list or None) index of original variable for realizations in
                pyramid levels (depends on input parameters given in
                deesse_input); if pyramid was used and output in pyramid required:
                    sim_pyramid_var_original_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_original_index[j][k]: index of the
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->originalVarIndexPyramidLevel[j])
                        (sim_pyramid_var_original_index[j] is None if
                        mpds_simoutput->originalVarIndexPyramidLevel[j] is NULL)
                (sim_pyramid_var_original_index is None otherwise)

        sim_pyramid_var_pyramid_index:
                (list or None) pyramid index of original variable for
                realizations in pyramid levels (depends on input parameters given
                in deesse_input); if pyramid was used and output in pyramid
                required:
                    sim_pyramid_var_pyramid_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_pyramid_index[j][k]: pyramid index of
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j])
                        (sim_pyramid_var_pyramid_index[j] is None if
                        mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]
                        is NULL)
                (sim_pyramid_var_pyramid_index is None otherwise)

        path:   (1-dimensional array of Img (class) of size nreal or None)
                    path[i]: path index map for the i-th realisation
                        (mpds_simoutput->outputPathIndexImage[0])
                    (path is None if mpds_simoutput->outputPathIndexImage is NULL)

        error:   (1-dimensional array of Img (class) of size nreal or None)
                    error[i]: error map for the i-th realisation
                        (mpds_simoutput->outputErrorImage[0])
                    (error is None if mpds_simoutput->outputErrorImage is NULL)

        tiGridNode:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiGridNode[i]: TI grid node index map for the i-th realisation
                        (mpds_simoutput->outputTiGridNodeIndexImage[0])
                    (tiGridNode is None if
                    mpds_simoutput->outputTiGridNodeIndexImage is NULL)

        tiIndex:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiIndex[i]: TI index map for the i-th realisation
                        (mpds_simoutput->outputTiIndexImage[0])
                    (tiIndex is None if
                    mpds_simoutput->outputTiIndexImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Initialization
    sim, sim_var_original_index = None, None
    sim_pyramid, sim_pyramid_var_original_index, sim_pyramid_var_pyramid_index = None, None, None
    path, error, tiGridNode, tiIndex = None, None, None, None
    nwarning, warnings = None, None

    if mpds_simoutput.nreal:
        nreal = mpds_simoutput.nreal

        if mpds_simoutput.nvarSimPerReal:
            # --- sim_var_original_index ---
            sim_var_original_index = np.zeros(mpds_simoutput.nvarSimPerReal, dtype='intc') # 'intc' for C-compatibility
            deesse.mpds_get_array_from_int_vector(mpds_simoutput.originalVarIndex, 0, sim_var_original_index)

            # ... also works ...
            # sim_var_original_index = np.asarray([deesse.int_array_getitem(mpds_simoutput.originalVarIndex, i) for i in range(mpds_simoutput.nvarSimPerReal)])
            # ...
            # ---

            # --- sim ---
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

            del(im)
            sim = np.asarray(sim).reshape(nreal)
            # ---

            if mpds_simoutput.npyramidLevel:
                npyramidLevel = mpds_simoutput.npyramidLevel

                nvarSimPerRealPyramidLevel = np.zeros(mpds_simoutput.npyramidLevel, dtype='intc') # 'intc' for C-compatibility
                deesse.mpds_get_array_from_int_vector(mpds_simoutput.nvarSimPerRealPyramidLevel, 0, nvarSimPerRealPyramidLevel)

                if np.sum(nvarSimPerRealPyramidLevel) > 0:
                    # --- sim_pyramid, sim_pyramid_var_original_index, sim_pyramid_var_pyramid_index ---
                    sim_pyramid = npyramidLevel*[None]
                    sim_pyramid_var_original_index = npyramidLevel*[None]
                    sim_pyramid_var_pyramid_index = npyramidLevel*[None]

                    for j in range(npyramidLevel):
                        if nvarSimPerRealPyramidLevel[j]:
                            # +++ sim_pyramid_var_original_index[j] +++
                            sim_pyramid_var_original_index[j] = np.zeros(nvarSimPerRealPyramidLevel[j], dtype='intc') # 'intc' for C-compatibility
                            iptr = deesse.intp_array_getitem(mpds_simoutput.originalVarIndexPyramidLevel, j)
                            deesse.mpds_get_array_from_int_vector(iptr, 0, sim_pyramid_var_original_index[j])

                            # # ... also works ...
                            # iptr = deesse.intp_array_getitem(mpds_simoutput.originalVarIndexPyramidLevel, j)
                            # sim_pyramid_var_original_index[j] = np.asarray([deesse.int_array_getitem(iptr, k) for k in range(nvarSimPerRealPyramidLevel[j])])
                            # # ...
                            # +++

                            # +++ sim_pyramid_var_pyramid_index[j] +++
                            sim_pyramid_var_pyramid_index[j] = np.zeros(nvarSimPerRealPyramidLevel[j], dtype='intc') # 'intc' for C-compatibility
                            iptr = deesse.intp_array_getitem(mpds_simoutput.pyramidIndexOfOriginalVarPyramidLevel, j)
                            deesse.mpds_get_array_from_int_vector(iptr, 0, sim_pyramid_var_pyramid_index[j])

                            # # ... also works ...
                            # iptr = deesse.intp_array_getitem(mpds_simoutput.pyramidIndexOfOriginalVarPyramidLevel, j)
                            # sim_pyramid_var_pyramid_index[j] = np.asarray([deesse.int_array_getitem(iptr, k) for k in range(nvarSimPerRealPyramidLevel[j])])
                            # # ...
                            # +++

                            # +++ sim_pyramid[j] +++
                            im_ptr = deesse.MPDS_IMAGEp_array_getitem(mpds_simoutput.outputSimImagePyramidLevel, j)
                            im = img_C2py(im_ptr)

                            nv = nvarSimPerRealPyramidLevel[j]
                            k = 0
                            sim_pyramid[j] = []
                            for i in range(nreal):
                                sim_pyramid[j].append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                               sx=im.sx, sy=im.sy, sz=im.sz,
                                               ox=im.ox, oy=im.oy, oz=im.oz,
                                               nv=nv, val=im.val[k:(k+nv),...],
                                               varname=im.varname[k:(k+nv)]))
                                k = k + nv

                            del(im)
                            sim_pyramid[j] = np.asarray(sim_pyramid[j]).reshape(nreal)
                            # +++
                    # ---

        if mpds_simoutput.nvarPathIndexPerReal:
            # --- path ---
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

            del(im)
            path = np.asarray(path).reshape(nreal)
            # ---

        if mpds_simoutput.nvarErrorPerReal:
            # --- error ---
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

            del(im)
            error = np.asarray(error).reshape(nreal)
            # ---

        if mpds_simoutput.nvarTiGridNodeIndexPerReal:
            # --- tiGridNode ---
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

            del(im)
            tiGridNode = np.asarray(tiGridNode).reshape(nreal)
            # ---

        if mpds_simoutput.nvarTiIndexPerReal:
            # --- tiIndex ---
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

            del(im)
            tiIndex = np.asarray(tiIndex).reshape(nreal)
            # ---

    # --- nwarning, warnings ---
    nwarning = mpds_progressMonitor.nwarning
    warnings = []
    if mpds_progressMonitor.nwarningNumber:
        tmp = np.zeros(mpds_progressMonitor.nwarningNumber, dtype='intc') # 'intc' for C-compatibility
        deesse.mpds_get_array_from_int_vector(mpds_progressMonitor.warningNumberList, 0, tmp)
        warningNumberList = np.asarray(tmp, dtype='int') # 'int' or equivalently 'int64'
        for iwarn in warningNumberList:
            warning_message = deesse.mpds_get_warning_message(int(iwarn)) # int() required!
            warning_message = warning_message.replace('\n', '')
            warnings.append(warning_message)
    # ---

    return {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'sim_pyramid':sim_pyramid, 'sim_pyramid_var_original_index':sim_pyramid_var_original_index, 'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
        'path':path, 'error':error, 'tiGridNode':tiGridNode, 'tiIndex':tiIndex,
        'nwarning':nwarning, 'warnings':warnings
        }
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseRun(deesse_input, add_data_point_to_mask=True, nthreads=-1, verbose=2):
    """
    Launches deesse.

    :param deesse_input:
                (DeesseInput (class)): deesse input parameter (python)

    :param add_data_point_to_mask:
                        (bool) indicating if grid cells out of the mask (simulated
                        part, if used) contains some data points (if present) are
                        added to the mask for the computation (this allows to
                        account for such data points, otherwise they are ignored);
                        at the end of the computation, the new mask cell are (if
                        any) are removed

    :param nthreads:
                (int) number of thread(s) to use for deesse (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the deesse run:
                    - 0: mininal display
                    - 1: only errors
                    - 2: version and warning(s) encountered
                    - 3 (or >2): version, progress, and warning(s) encountered

    :return deesse_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'sim_pyramid':sim_pyramid,
             'sim_pyramid_var_original_index':sim_pyramid_var_original_index,
             'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
             'path':path,
             'error':error,
             'tiGridNode':tiGridNode,
             'tiIndex':tiIndex,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = deesse_input.nrealization:

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_simoutput->outputSimImage[0])
                    (sim is None if mpds_simoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_simoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_simoutput->originalVarIndex is NULL)

        sim_pyramid:
                (list or None) realizations in pyramid levels (depends on input
                parameters given in deesse_input); if pyramid was used and output
                in pyramid required:
                    sim_pyramid[j]:
                        (1-dimensional array of Img (class) of size nreal or None)
                        sim_pyramid[j][i]: i-th realisation in pyramid level of
                            index j+1, k-th variable stored refers to
                                - the original variable
                                    sim_pyramid_var_original_index[j][k]
                                - and pyramid index
                                    sim_pyramid_var_pyramid_index[j][k]
                            (get from
                            mpds_simoutput->outputSimImagePyramidLevel[j])
                        (sim_pyramid[j] is None if
                        mpds_simoutput->outputSimImagePyramidLevel[j] is NULL)
                (sim_pyramid is None otherwise)

        sim_pyramid_var_original_index:
                (list or None) index of original variable for realizations in
                pyramid levels (depends on input parameters given in
                deesse_input); if pyramid was used and output in pyramid required:
                    sim_pyramid_var_original_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_original_index[j][k]: index of the
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->originalVarIndexPyramidLevel[j])
                        (sim_pyramid_var_original_index[j] is None if
                        mpds_simoutput->originalVarIndexPyramidLevel[j] is NULL)
                (sim_pyramid_var_original_index is None otherwise)

        sim_pyramid_var_pyramid_index:
                (list or None) pyramid index of original variable for
                realizations in pyramid levels (depends on input parameters given
                in deesse_input); if pyramid was used and output in pyramid
                required:
                    sim_pyramid_var_pyramid_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_pyramid_index[j][k]: pyramid index of
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j])
                        (sim_pyramid_var_pyramid_index[j] is None if
                        mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]
                        is NULL)
                (sim_pyramid_var_pyramid_index is None otherwise)

        path:   (1-dimensional array of Img (class) of size nreal or None)
                    path[i]: path index map for the i-th realisation
                        (mpds_simoutput->outputPathIndexImage[0])
                    (path is None if mpds_simoutput->outputPathIndexImage is NULL)

        error:   (1-dimensional array of Img (class) of size nreal or None)
                    error[i]: error map for the i-th realisation
                        (mpds_simoutput->outputErrorImage[0])
                    (error is None if mpds_simoutput->outputErrorImage is NULL)

        tiGridNode:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiGridNode[i]: TI grid node index map for the i-th realisation
                        (mpds_simoutput->outputTiGridNodeIndexImage[0])
                    (tiGridNode is None if
                    mpds_simoutput->outputTiGridNodeIndexImage is NULL)

        tiIndex:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiIndex[i]: TI index map for the i-th realisation
                        (mpds_simoutput->outputTiIndexImage[0])
                    (tiIndex is None if
                    mpds_simoutput->outputTiIndexImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    fname = 'deesseRun'

    if not deesse_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesse input')
        return None

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if deesse_input.mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(deesse_input.mask)
        # Add cell to mask if needed
        for ps in deesse_input.dataPointSet:
            im_tmp = img.imageFromPoints(ps.val[:3].T,
                    nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                    sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                    ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                    indicator_var=True)
            deesse_input.mask = 1.0*np.any((im_tmp.val[0], deesse_input.mask), axis=0)
            del (im_tmp)

    if verbose >= 2:
        print('DeeSse running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesse...

    # Convert deesse input from python to C
    try:
        mpds_siminput = deesse_input_py2C(deesse_input)
    except:
        print(f'ERROR ({fname}): unable to convert deesse input from python to C...')
        return None

    if mpds_siminput is None:
        print(f'ERROR ({fname}): unable to convert deesse input from python to C...')
        return None

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
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor4_ptr

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

    if deesse_input.mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        for im in deesse_output['sim']:
            im.val[:, mask_original==0.0] = np.nan

    if verbose >= 2 and deesse_output:
        print('DeeSse run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and deesse_output and deesse_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(deesse_output['nwarning']))
        for i, warning_message in enumerate(deesse_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return deesse_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseRun_mp(deesse_input, add_data_point_to_mask=True, nproc=None, nthreads_per_proc=None, verbose=2):
    """
    Launches deesse through multiple processes.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function deesseRun],
        - the set of realizations (specified by deesse_input.nrealization) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param deesse_input:
                (DeesseInput (class)): deesse input parameter (python)

    :param add_data_point_to_mask:
                        (bool) indicating if grid cells out of the mask (simulated
                        part, if used) contains some data points (if present) are
                        added to the mask for the computation (this allows to
                        account for such data points, otherwise they are ignored);
                        at the end of the computation, the new mask cell are (if
                        any) are removed
    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count()), and
                    nreal is the number of realization (deesse_input.nrealization)

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: mininal display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return deesse_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'sim_pyramid':sim_pyramid,
             'sim_pyramid_var_original_index':sim_pyramid_var_original_index,
             'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
             'path':path,
             'error':error,
             'tiGridNode':tiGridNode,
             'tiIndex':tiIndex,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = deesse_input.nrealization:

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_simoutput->outputSimImage[0])
                    (sim is None if mpds_simoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_simoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_simoutput->originalVarIndex is NULL)

        sim_pyramid:
                (list or None) realizations in pyramid levels (depends on input
                parameters given in deesse_input); if pyramid was used and output
                in pyramid required:
                    sim_pyramid[j]:
                        (1-dimensional array of Img (class) of size nreal or None)
                        sim_pyramid[j][i]: i-th realisation in pyramid level of
                            index j+1, k-th variable stored refers to
                                - the original variable
                                    sim_pyramid_var_original_index[j][k]
                                - and pyramid index
                                    sim_pyramid_var_pyramid_index[j][k]
                            (get from
                            mpds_simoutput->outputSimImagePyramidLevel[j])
                        (sim_pyramid[j] is None if
                        mpds_simoutput->outputSimImagePyramidLevel[j] is NULL)
                (sim_pyramid is None otherwise)

        sim_pyramid_var_original_index:
                (list or None) index of original variable for realizations in
                pyramid levels (depends on input parameters given in
                deesse_input); if pyramid was used and output in pyramid required:
                    sim_pyramid_var_original_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_original_index[j][k]: index of the
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->originalVarIndexPyramidLevel[j])
                        (sim_pyramid_var_original_index[j] is None if
                        mpds_simoutput->originalVarIndexPyramidLevel[j] is NULL)
                (sim_pyramid_var_original_index is None otherwise)

        sim_pyramid_var_pyramid_index:
                (list or None) pyramid index of original variable for
                realizations in pyramid levels (depends on input parameters given
                in deesse_input); if pyramid was used and output in pyramid
                required:
                    sim_pyramid_var_pyramid_index[j]:
                        (1-dimensional array of ints or None)
                        sim_pyramid_var_pyramid_index[j][k]: pyramid index of
                            original variable (given in deesse_input) of the k-th
                            variable stored in sim_pyramid[j][i], for any i
                            (array of length array of length sim_pyramid[j][*].nv,
                            get from
                            mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j])
                        (sim_pyramid_var_pyramid_index[j] is None if
                        mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]
                        is NULL)
                (sim_pyramid_var_pyramid_index is None otherwise)

        path:   (1-dimensional array of Img (class) of size nreal or None)
                    path[i]: path index map for the i-th realisation
                        (mpds_simoutput->outputPathIndexImage[0])
                    (path is None if mpds_simoutput->outputPathIndexImage is NULL)

        error:   (1-dimensional array of Img (class) of size nreal or None)
                    error[i]: error map for the i-th realisation
                        (mpds_simoutput->outputErrorImage[0])
                    (error is None if mpds_simoutput->outputErrorImage is NULL)

        tiGridNode:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiGridNode[i]: TI grid node index map for the i-th realisation
                        (mpds_simoutput->outputTiGridNodeIndexImage[0])
                    (tiGridNode is None if
                    mpds_simoutput->outputTiGridNodeIndexImage is NULL)

        tiIndex:
                (1-dimensional array of Img (class) of size nreal or None)
                    tiIndex[i]: TI index map for the i-th realisation
                        (mpds_simoutput->outputTiIndexImage[0])
                    (tiIndex is None if
                    mpds_simoutput->outputTiIndexImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    fname = 'deesseRun_mp'

    if not deesse_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesse input')
        return None

    if deesse_input.nrealization <= 1:
        if verbose > 0:
            print('NOTE: number of realization does not exceed 1: launching deesseRun...')
        nthreads = nthreads_per_proc
        if nthreads is None:
            nthreads = -1
        deesse_output = deesseRun(deesse_input, add_data_point_to_mask=add_data_point_to_mask, nthreads=nthreads, verbose=verbose)
        return deesse_output

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, deesse_input.nrealization), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), deesse_input.nrealization), 1)
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

    if deesse_input.mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(deesse_input.mask)
        # Add cell to mask if needed
        for ps in deesse_input.dataPointSet:
            im_tmp = img.imageFromPoints(ps.val[:3].T,
                    nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                    sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                    ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                    indicator_var=True)
            deesse_input.mask = 1.0*np.any((im_tmp.val[0], deesse_input.mask), axis=0)
            del (im_tmp)

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(deesse_input.nrealization, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('DeeSse running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesse...

    # Initialize deesse input for each process
    deesse_input_proc = [copy.copy(deesse_input) for i in range(nproc)]
    init_seed = deesse_input.seed

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i, input in enumerate(deesse_input_proc):
        # Adapt deesse input for i-th process
        input.nrealization = real_index_proc[i+1] - real_index_proc[i]
        input.seed = init_seed + int(real_index_proc[i]) * input.seedIncrement
        input.outputReportFileName = input.outputReportFileName + f'.{i}'
        if i==0:
            verb = min(verbose, 1) # allow to print error for process i
        else:
            verb = 0
        # Launch deesse (i-th process)
        out_pool.append(pool.apply_async(deesseRun, args=(input, False, nth, verb)))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    deesse_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in deesse_output_proc]):
        return None

    sim, sim_var_original_index = None, None
    sim_pyramid, sim_pyramid_var_original_index, sim_pyramid_var_pyramid_index = None, None, None
    path, error, tiGridNode, tiIndex = None, None, None, None
    nwarning, warnings = None, None

    # Gather results from every process
    # sim
    sim = np.hstack([out['sim'] for out in deesse_output_proc])
    # ... remove None entries
    sim = sim[[x is not None for x in sim]]
    # ... set to None if every entry is None
    if np.all([x is None for x in sim]):
        sim = None

    # sim_var_original_index
    sim_var_original_index = deesse_output_proc[0]['sim_var_original_index']

    # sim_pyramid
    nlevel = 0
    if deesse_output_proc[0]['sim_pyramid'] is not None:
        nlevel = len(deesse_output_proc[0]['sim_pyramid'])
        sim_pyramid = []
        for j in range(nlevel):
            # set sim_lev (that will be append to sim_pyramid)
            sim_lev = []
            for out in deesse_output_proc:
                if out['sim_pyramid'] is not None:
                    sim_lev.append(out['sim_pyramid'][j])
            sim_lev = np.hstack(sim_lev)
            # ... remove None entries
            sim_lev = sim_lev[[x is not None for x in sim_lev]]
            # ... set to None if every entry is None
            if np.all([x is None for x in sim_lev]):
                sim_lev = None
            # ... append to sim_pyramid
            sim_pyramid.append(sim_lev)

    # sim_pyramid_var_original_index
    sim_pyramid_var_original_index = deesse_output_proc[0]['sim_pyramid_var_original_index']

    # sim_pyramid_var_pyramid_index
    sim_pyramid_var_pyramid_index = deesse_output_proc[0]['sim_pyramid_var_pyramid_index']

    # path
    path = np.hstack([out['path'] for out in deesse_output_proc])
    # ... remove None entries
    path = path[[x is not None for x in path]]
    # ... set to None if every entry is None
    if np.all([x is None for x in path]):
        path = None

    # error
    error = np.hstack([out['error'] for out in deesse_output_proc])
    # ... remove None entries
    error = error[[x is not None for x in error]]
    # ... set to None if every entry is None
    if np.all([x is None for x in error]):
        error = None

    # tiGridNode
    tiGridNode = np.hstack([out['tiGridNode'] for out in deesse_output_proc])
    # ... remove None entries
    tiGridNode = tiGridNode[[x is not None for x in tiGridNode]]
    # ... set to None if every entry is None
    if np.all([x is None for x in tiGridNode]):
        tiGridNode = None

    # tiIndex
    tiIndex = np.hstack([out['tiIndex'] for out in deesse_output_proc])
    # ... remove None entries
    tiIndex = tiIndex[[x is not None for x in tiIndex]]
    # ... set to None if every entry is None
    if np.all([x is None for x in tiIndex]):
        tiIndex = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in deesse_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in deesse_output_proc])))

    # Adjust variable names
    ndigit = deesse.MPDS_NB_DIGIT_FOR_REALIZATION_NUMBER
    if sim is not None:
        for i in range(deesse_input.nrealization):
            for k in range(sim[i].nv):
                sim[i].varname[k] = sim[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if sim_pyramid is not None:
        for j in range(nlevel):
            if sim_pyramid[j] is not None:
                for i in range(deesse_input.nrealization):
                    for k in range(sim_pyramid[j][i].nv):
                        sim_pyramid[j][i].varname[k] = sim_pyramid[j][i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if path is not None:
        for i in range(deesse_input.nrealization):
            for k in range(path[i].nv):
                path[i].varname[k] = path[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if error is not None:
        for i in range(deesse_input.nrealization):
            for k in range(error[i].nv):
                error[i].varname[k] = error[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if tiGridNode is not None:
        for i in range(deesse_input.nrealization):
            for k in range(tiGridNode[i].nv):
                tiGridNode[i].varname[k] = tiGridNode[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if tiIndex is not None:
        for i in range(deesse_input.nrealization):
            for k in range(tiIndex[i].nv):
                tiIndex[i].varname[k] = tiIndex[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'

    if deesse_input.mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        for im in sim:
            im.val[:, mask_original==0.0] = np.nan

    deesse_output = {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'sim_pyramid':sim_pyramid, 'sim_pyramid_var_original_index':sim_pyramid_var_original_index, 'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
        'path':path, 'error':error, 'tiGridNode':tiGridNode, 'tiIndex':tiIndex,
        'nwarning':nwarning, 'warnings':warnings
        }

    if verbose >= 2 and deesse_output:
        print('DeeSse run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and deesse_output and deesse_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(deesse_output['nwarning']))
        for i, warning_message in enumerate(deesse_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return deesse_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def exportDeesseInput(
        deesse_input,
        dirname='input_ascii',
        fileprefix='ds',
        endofline='\n',
        verbose=1):
    """
    Exports deesse input as ASCII files (in the directory named <dirname>).
    The command line version of deesse can then be launched from the directory
    <dirname> by using the generated ASCII files.

    :param deesse_input:    (DeesseInput class) deesse input - python
    :param dirname:         (string) name of the directory in which the files
                                will be written; if not existing, it will be
                                created;
                                WARNING: the generated files might erase already
                                existing ones!
    :param fileprefix:      (string) prefix for generated files, the main input
                                file will be <dirname>/<fileprefix>.in
    :param endofline:       (string) end of line string to be used for the deesse
                                input file
    :param verbose:         (int) indicates which degree of detail is used when
                                writing comments in the deesse input file
                                - 0: no comment
                                - 1: basic comments
                                - 2: detailed comments
    """

    fname = 'exportDeesseInput'

    if not deesse_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesse input')
        return None

    # Create ouptut directory if needed
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Convert deesse input from python to C
    try:
        mpds_siminput = deesse_input_py2C(deesse_input)
    except:
        print(f'ERROR ({fname}): unable to convert deesse input from python to C...')
        return None

    if mpds_siminput is None:
        print(f'ERROR ({fname}): unable to convert deesse input from python to C...')
        return None

    err = deesse.MPDSExportSimInput( mpds_siminput, dirname, fileprefix, endofline, verbose)

    if err:
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)

    # Free memory on C side: deesse input
    deesse.MPDSFreeSimInput(mpds_siminput)
    #deesse.MPDSFree(mpds_siminput)
    deesse.free_MPDS_SIMINPUT(mpds_siminput)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def importDeesseInput(filename, dirname='.'):
    """
    Imports deesse input from ASCII files, used for command line version of
    deesse (from the directory named <dirname>).

    :param filename:        (string) name of the general input ASCII file
                                (without path) used for the command line
                                version of deesse
    :param dirname:         (string) name of the directory in which the input
                                ASCII files are stored (and from which the
                                command line version of deesse would be
                                launched)

    :param deesse_input:    (DeesseInput class) deesse input - python
    """

    fname = 'importDeesseInput'

    # Check directory
    if not os.path.isdir(dirname):
        print(f'ERROR ({fname}): directory does not exist')
        return None

    # Check file
    if not os.path.isfile(os.path.join(dirname, filename)):
        print(f'ERROR ({fname}): input file does not exist')
        return None

    # Get current working directory
    cwd = os.getcwd()

    # Change directory
    os.chdir(dirname)

    try:
        # Initialization a double pointer onto MPDS_SIMINPUT
        mpds_siminputp = deesse.new_MPDS_SIMINPUTp()

        # Import
        deesse.MPDSImportSimInput(filename, mpds_siminputp)

        # Retrieve structure
        mpds_siminput = deesse.MPDS_SIMINPUTp_value(mpds_siminputp)

        # Convert deesse input from C to python
        deesse_input = deesse_input_C2py(mpds_siminput)

    except:
        deesse_input = None

    if deesse_input is None:
        print(f'ERROR ({fname}): unable to import deesse input from ASCII files...')

    # Change directory (to initial working directory)
    os.chdir(cwd)

    return deesse_input
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgPyramidImage(
        input_image,
        operation='reduce',
        kx=None,
        ky=None,
        kz=None,
        w0x=None,
        w0y=None,
        w0z=None,
        minWeight=None,
        nthreads=-1):
    """
    Computes the Gaussian (pyramid) reduction or expansion of the input image.
    This function applies the Gaussian pyramid reduction or expansion to all
    variables (treated as continuous) of the input image, and returns an output
    image with the same number of variables, whose the names are the same as the
    variables of the input image, followed by a suffix the suffix "_GPred" (resp.
    "_GPexp") if reduction (resp. expansion) is applied. The grid (support) of
    the output image is derived from the Gaussian pyramid operation.
    The Gaussian operation consists in applying a weighted moving average using a
    Gaussian-like kernel (or filter) of size (2*kx + 1) x (2*ky + 1) x (2*kz + 1)
    [see parameters below], while in the output image grid the number of cells
    along x, y, z-axis will be divided (resp. multiplied) by a factor (of about)
    kx, ky, kz respectively if reduction (resp. expansion) is applied.

    :param input_image:
                    (Img class) input image
    :param operation:
                    (string) operation to apply, either 'reduce' (default)
                        or 'expand'

    :param kx, ky, kz:
                    (ints) reduction step along x, y, z-direction:
                            k[x|y|z] = 0: nothing is done, same dimension
                                            in output
                            k[x|y|z] = 1: same dimension in output (with
                                            weighted average over 3 nodes)
                            k[x|y|z] = 2: classical gaussian pyramid
                            k[x|y|z] > 2: generalized gaussian pyramid
                        By defaut (None), the reduction step will be set to 2 in
                        directions where the input image grid has more than one
                        cell, and to 0 in other directions

    :param w0x, w0y, w0z:
                    (floats) weight for central cell in the kernel (filter) when
                        computing average during Gaussian pyramid operation in
                        x,y,z-direction; specifying None (default) or a negative
                        value, the default weight derived from proportionality
                        with Gaussian weights (binomial coefficients) will be
                        used; specifying a positive value or zero implies to
                        explicitly set the weight to that value

   :param minWeight:
                    (float) minimal weight on informed cells within the filter to
                        define output value: when applying the moving weighted
                        average, if the total weight on informed cells within the
                        kernel (filter) is less than minWeight, undefined value
                        (np.nan) is set as output value, otherwise the weigted
                        average is set; specifiying None (default) or a negative
                        value, a default minimal weight will be used; specifying
                        a positive value or zero implies to explicitly set the
                        minimal weight to that value;
                        Note: the default minimal weight is
                        geone.deesse_core.deesse.MPDS_GAUSSIAN_PYRAMID_RED_TOTAL_WEIGHT_MIN
                        for reduction, and
                        geone.deesse_core.deesse.MPDS_GAUSSIAN_PYRAMID_EXP_TOTAL_WEIGHT_MIN
                        for expansion

    :param nthreads:
                    (int) number of thread(s) to use for program (C),
                        (nthreads = -n <= 0: for maximal number of threads
                        except n, but at least 1)

    :return output_image:   (Img class) output image
    """

    fname = 'imgPyramidImage'

    # --- Check
    if operation not in ('reduce', 'expand'):
        print(f"ERROR ({fname}): unknown 'operation'")
        return None

    # --- Prepare parameters
    if kx is None:
        if input_image.nx == 1:
            kx = 0
        else:
            kx = 2
    else:
        kx = int(kx) # ensure int type

    if ky is None:
        if input_image.ny == 1:
            ky = 0
        else:
            ky = 2
    else:
        ky = int(ky) # ensure int type

    if kz is None:
        if input_image.nz == 1:
            kz = 0
        else:
            kz = 2
    else:
        kz = int(kz) # ensure int type

    if w0x is None:
        w0x = -1.0
    else:
        w0x = float(w0x) # ensure float type

    if w0y is None:
        w0y = -1.0
    else:
        w0y = float(w0y) # ensure float type

    if w0z is None:
        w0z = -1.0
    else:
        w0z = float(w0z) # ensure float type

    if minWeight is None:
        if operation == 'reduce':
            minWeight = deesse.MPDS_GAUSSIAN_PYRAMID_RED_TOTAL_WEIGHT_MIN
        else: # 'expand'
            minWeight = deesse.MPDS_GAUSSIAN_PYRAMID_EXP_TOTAL_WEIGHT_MIN
    else:
        minWeight = float(minWeight) # ensure float type

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output image "in C"
    output_image_c = deesse.malloc_MPDS_IMAGE()
    deesse.MPDSInitImage(output_image_c)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    # --- Compute pyramid (launch C code)
    if operation == 'reduce':
        err = deesse.MPDSOMPImagePyramidReduce(input_image_c, output_image_c, kx, ky, kz, w0x, w0y, w0z, minWeight, nth)
    elif operation == 'expand':
        err = deesse.MPDSOMPImagePyramidExpand(input_image_c, output_image_c, kx, ky, kz, w0x, w0y, w0z, minWeight, nth)
    else:
        print(f"ERROR ({fname}): 'operation' not valid")
        return None

    # --- Retrieve output image "in python"
    if err:
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        output_image = None
    else:
        output_image = img_C2py(output_image_c)

    # Free memory on C side: input_image_c
    deesse.MPDSFreeImage(input_image_c)
    #deesse.MPDSFree (input_image_c)
    deesse.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    deesse.MPDSFreeImage(output_image_c)
    #deesse.MPDSFree (output_image_c)
    deesse.free_MPDS_IMAGE(output_image_c)

    return output_image
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgCategoricalToContinuous(
        input_image,
        varInd=None,
        xConnectFlag=None,
        yConnectFlag=None,
        zConnectFlag=None,
        nthreads=-1):
    """
    Transforms the desired variable(s), considered as categorical, from the input
    image, into "continuous" variable(s) (with values in [0, 1]), and returns the
    corresponding output image. The transformation for a variable with n
    categories is done such that:
        - each category in input will correspond to a distinct output value
            in {i/(n-1), i=0, ..., n-1}
        - the output values are set such that "closer values correspond to better
            connected (more contact btw.) categories"
        - this is the transformation done by deesse when pyramid is used with
            pyramid type ('pyramidType') set to 'categorical_to_continuous'.

    :param input_image: (Img class) input image

    :param varInd:      (sequence of ints or int or None) index-es of the
                            variables for which the transformation has to be done

    :param xConnectFlag, yConnectFlag, zConnectFlag:
                        (bool) flag indicating if the connction (contact btw.)
                            categories are accounted for in x, y, z-direction
                            (corresponding flag set to True) or not (corresponding
                            flag set to False); By default (None), these flag will
                            be set to True, provided that the number of cells in
                            the corresponding direction is greater than 1

    :param nthreads:    (int) number of thread(s) to use for program (C),
                            (nthreads = -n <= 0: for maximal number of threads
                            except n, but at least 1)

    :return output_image:   (Img class) output image
    """

    fname = 'imgCategoricalToContinuous'

    # --- Check
    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if np.sum([iv in range(input_image.nv) for iv in varInd]) != len(varInd):
            print(f'ERROR ({fname}): invalid index-es')
            return None
    else:
        varInd = np.arange(input_image.nv)

    # --- Prepare parameters
    if xConnectFlag is None:
        if input_image.nx == 1:
            xConnectFlag = False
        else:
            xConnectFlag = True

    if yConnectFlag is None:
        if input_image.ny == 1:
            yConnectFlag = False
        else:
            yConnectFlag = True

    if zConnectFlag is None:
        if input_image.nz == 1:
            zConnectFlag = False
        else:
            zConnectFlag = True

    # Initialize output image
    output_image = img.copyImg(input_image)

    # Initialize value index
    val_index = np.zeros(input_image.nxyz(), dtype='intc')

    # --- Initialize vector in C
    val_index_c = deesse.new_int_array(int(val_index.size))

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    ok = True # to intercept error during for loop...
    for ind in varInd:
        # Get vector of values of the variale of index ind from input image
        vptr = input_image.val[ind].reshape(-1)

        unique_val = input_image.get_unique_one_var(ind)
        for i, v in enumerate(unique_val):
            val_index[vptr==v] = i

        # ... set index -1 for nan
        val_index[np.isnan(vptr)] = -1

        n = len(unique_val)

        # --- Set vectors in C
        deesse.mpds_set_int_vector_from_array(val_index_c, 0, val_index)

        to_new_index_c = deesse.new_int_array(n)
        to_initial_index_c = deesse.new_int_array(n)

        # --- Compute index correspondence (launch C code)
        err = deesse.MPDSOMPGetImageOneVarNewValueIndexOrder(
            input_image.nx, input_image.ny, input_image.nz,
            n, val_index_c,
            to_new_index_c, to_initial_index_c,
            int(xConnectFlag), int(yConnectFlag), int(zConnectFlag),
            nth)

        # --- Retrieve vector to_new_index and to_initial_index "in python"
        if err:
            err_message = deesse.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
            ok = False
        else:
            to_new_index = np.zeros(n, dtype='intc') # 'intc' for C-compatibility
            deesse.mpds_get_array_from_int_vector (to_new_index_c, 0, to_new_index)

            # not used!
            # to_initial_index = np.zeros(n, dtype='intc') # 'intc' for C-compatibility
            # deesse.mpds_get_array_from_int_vector (to_initial_index_c, 0, to_initial_index)

            # set new values
            r = 1./max(n-1, 1)
            vptr = output_image.val[ind].reshape(-1) # pointer !
            for i in range(n):
                vptr[val_index==i] = r * to_new_index[i]

        # Free memory on C side
        deesse.delete_int_array(to_new_index_c)
        deesse.delete_int_array(to_initial_index_c)

        if not ok:
            break

    # Free memory on C side
    deesse.delete_int_array(val_index_c)

    return output_image
# ----------------------------------------------------------------------------

##### Additional stuff for deesseX #####

# ============================================================================
class DeesseXInputSectionPath(object):
    """
    Defines section mode and path for cross-simulation (deesseX):
        sectionMode:
            (string) section mode, defining which type of section will be
                simulated alternately, possible values:
                    'section_xy_xz_yz',
                    'section_xy_yz_xz',
                    'section_xz_xy_yz',
                    'section_xz_yz_xy',
                    'section_yz_xy_xz',
                    'section_yz_xz_xy',
                    'section_xy_xz',
                    'section_xz_xy',
                    'section_xy_yz',
                    'section_yz_xy',
                    'section_xz_yz',
                    'section_yz_xz',
                    'section_xy_z',
                    'section_z_xy',
                    'section_xz_y',
                    'section_y_xz',
                    'section_yz_x',
                    'section_x_yz',
                    'section_x_y_z',
                    'section_x_z_y',
                    'section_y_x_z',
                    'section_y_z_x',
                    'section_z_x_y',
                    'section_z_y_x',
                    'section_x_y',
                    'section_y_x',
                    'section_x_z',
                    'section_z_x',
                    'section_y_z',
                    'section_z_y'
                'section_<t_1>_<t_2>[_<t_3>]': means that simulation in 2D
                    (resp. 1D) will be done alternately in sections parallel to
                    the plane (resp. axis) given in the string '<t_i>';
                notes:
                    - the order can matter (depending on sectionPathMode)
                    - the mode involving only two 1D axis as section (i.e.
                        'section_x_y' to 'section_z_y') can be used for a
                        two-dimensional simulation grid

        sectionPathMode:
            (string) section path mode, defining the section path, i.e. the
                succession of simulated section, possible values:
                - 'section_path_random': random section path

                - 'section_path_pow_2': indexes (of cells locating the section)
                    in the orthogonal direction of the sections, are chosen as
                    decreasing power of 2 (dealing alternately with each section
                    orientation in the order given by sectionMode)

                - 'section_path_subdiv': succession of sections is defined as:
                    (a) for each section orientation (in the order given by
                        sectionMode), the section corresponding to the most left
                        border (containing the origin) of the simulation grid is
                        selected
                    (b) let minspaceX, minspaceY, minspaceZ (see parameters
                        below), the minimal space (or step) in number of cells
                        along x, y, z axis resp. between two successive sections
                        of the same type and orthogonal to x, y, z axis resp.:
                        (i) for each section orientation (in the order given by
                            sectionMode): the section(s) corresponding to the
                            most right border (face or edge located at one largest
                            index in the corresponding direction) of the
                            simulation grid is selected, provided that the space
                            (step) with the previous section (selected in (a))
                            satisfies the minimal space in the relevant direction
                        (ii) for each section orientation (in the order given by
                            sectionMode): the sections between the borders are
                            selected, such that they are regularly spaced along
                            any direction (with a difference of at most one cell)
                            and such that the minimal space is satisfied (i.e.
                            the number of cell from one section to the next one
                            is at least equal to corresponding parameter
                            minspaceX, minspaceY or minspaceZ)
                        (iii) for each section orientation (in the order given by
                            sectionMode): if in step (i) the right border was not
                            selected (due to a space less than the minimal space
                            paremeter(s)), then it is selected here
                        note that at the end of step (b), there are at least two
                        sections of same type along any axis direction (having
                        more than one cell in the entire simulation grid)
                    (c) next, the middle sections (along each direction) between
                        each pair of consecutive sections already selected are
                        selected, until the entire simulation grid is filled,
                        following one of the two methods (see parameter
                        balancedFillingFlag below):
                        - if balancedFillingFlag is False:
                            considering alternately each section orientation, in
                            the order given by sectionMode,
                        - if balancedFillingFlag is True:
                            choosing the axis direction (x, y, or z) for which
                            the space (in number of cells) between two
                            consecutive sections already selected is the largest,
                            then selecting the section orientation(s) (among
                            those given by sectionMode) orthogonal to that
                            direction, and considering the middle sections with
                            respect to that direction

                    - 'section_path_manual':  succession of sections explicitly
                        given (see nsection, sectionType and sectionLoc below)

        minSpaceX:
            used iff sectionPathMode is set to 'section_path_subdiv',
            (float) minimal space in number of cells along x direction,
                in step (b) above;
                note:
                    - if minSpaceX > 0: use as it in step (b)
                    - if minSpaceX = 0: ignore (skip) step (b,ii) for x direction
                    - if minSpaceX < 0: this parameter is automatically computed,
                        and defined as the "range" in the x direction computed
                        from the training image(s) used in section(s) including
                        the x direction

        minSpaceY:
            (float) same as minSpaceX, but in y direction

        minSpaceZ:
            (float) same as minSpaceZ, but in z direction

        balancedFillingFlag:
            used iff sectionPathMode is set to 'section_path_subdiv',
            (bool) boolean flag defining the method used in step (c) above

        nsection:
            used iff sectionPathMode is set to 'section_path_manual',
            (int) number of section(s) to be simulated at toatl [sections
                (2D and/or 1D)]
                note: a partial filling of the simulation grid can be considered

        sectionType:
            used iff sectionPathMode is set to 'section_path_manual',
            (sequence of ints of length nsection) indexes identifying the type
                of sections:
                - sectionType[i]: type id of the i-th simulated section,
                    0 <= i < nsectionm, with:
                        id = 0: xy section (2D)
                        id = 1: xz section (2D)
                        id = 2: yz section (2D)
                        id = 3: z section (1D)
                        id = 4: y section (1D)
                        id = 5: x section (1D)

        sectionLoc:
            used iff sectionPathMode is set to 'section_path_manual',
            (sequence of ints of length nsection) indexes location of sections:
                - sectionLoc[i]: location of the i-th simulated section,
                    0 <= i < nsection, with:
                    - if sectionType[i] = 0 (xy), then
                        sectionLoc[i]=k in {0, ..., nz-1},
                        k is the index location along x axis
                    - if sectionType[i] = 1 (xz), then
                        sectionLoc[i]=k in {0, ..., ny-1},
                        k is the index location along y axis
                    - if sectionType[i] = 2 (yz), then
                        sectionLoc[i]=k in {0, ..., nx-1},
                        k is the index location along z axis
                    - if sectionType[i] = 3 (z), then
                        sectionLoc[i]=k in {0, ..., nx*ny-1},
                        (k%nx, k//nx) is the two index locations in xy section
                    - if sectionType[i] = 4 (y), then
                        sectionLoc[i]=k in {0, ..., nx*nz-1},
                        (k%nx, k//nx) is the two index locations in xz section
                    - if sectionType[i] = 5 (x), then
                        sectionLoc[i]=k in {0, ..., ny*nz-1},
                        (k%ny, k//ny) is the two index locations in yz section
                    and with nx, ny, nz the number of nodes in the entire
                    simulation grid along x, y, z axis respectively
    """

    def __init__(self,
                 sectionMode='section_xz_yz',
                 sectionPathMode='section_path_subdiv',
                 minSpaceX=None,
                 minSpaceY=None,
                 minSpaceZ=None,
                 balancedFillingFlag=True,
                 nsection=0,
                 sectionType=None,
                 sectionLoc=None):
        # sectionMode
        sectionMode_avail = (
            'section_xy_xz_yz',
            'section_xy_yz_xz',
            'section_xz_xy_yz',
            'section_xz_yz_xy',
            'section_yz_xy_xz',
            'section_yz_xz_xy',
            'section_xy_xz',
            'section_xz_xy',
            'section_xy_yz',
            'section_yz_xy',
            'section_xz_yz',
            'section_yz_xz',
            'section_xy_z',
            'section_z_xy',
            'section_xz_y',
            'section_y_xz',
            'section_yz_x',
            'section_x_yz',
            'section_x_y_z',
            'section_x_z_y',
            'section_y_x_z',
            'section_y_z_x',
            'section_z_x_y',
            'section_z_y_x',
            'section_x_y',
            'section_y_x',
            'section_x_z',
            'section_z_x',
            'section_y_z',
            'section_z_y'
        )

        fname = 'DeesseXInputSectionPath'

        if sectionMode not in sectionMode_avail:
            print(f'ERROR ({fname}): unknown sectionMode')
            return None

        self.sectionMode = sectionMode

        # sectionPathMode
        sectionPathMode_avail = (
            'section_path_random',
            'section_path_pow_2',
            'section_path_subdiv',
            'section_path_manual'
        )
        if sectionPathMode not in sectionPathMode_avail:
            print(f'ERROR ({fname}): unknown sectionPathMode')
            return None

        self.sectionPathMode = sectionPathMode

        if self.sectionPathMode == 'section_path_subdiv':
            # minSpaceX, minSpaceY, minSpaceZ
            if minSpaceX is None:
                self.minSpaceX = -1
            else:
                self.minSpaceX = minSpaceX

            if minSpaceY is None:
                self.minSpaceY = -1
            else:
                self.minSpaceY = minSpaceY

            if minSpaceZ is None:
                self.minSpaceZ = -1
            else:
                self.minSpaceZ = minSpaceZ

            # balancedFillingFlag
            self.balancedFillingFlag = balancedFillingFlag

        else:
            self.minSpaceX = 0.0 # unused
            self.minSpaceY = 0.0 # unused
            self.minSpaceZ = 0.0 # unused
            self.balancedFillingFlag = False # unused

        if self.sectionPathMode == 'section_path_manual':
            self.nsection = nsection
            if nsection > 0:
                try:
                    self.sectionType = np.asarray(sectionType, dtype='int').reshape(nsection)
                except:
                    print(f'ERROR ({fname}): field "sectionType"...')
                    return None
                try:
                    self.sectionLoc = np.asarray(sectionLoc, dtype='int').reshape(nsection)
                except:
                    print(f'ERROR ({fname}): field "sectionLoc"...')
                    return None
            else:
                self.sectionType = None
                self.sectionLoc = None

        else:
            self.nsection = 0       # unused
            self.sectionType = None # unused
            self.sectionLoc = None  # unused

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        out = '*** DeesseXInputSectionPath object ***'
        out = out + '\n' + 'sectionMode = {0.sectionMode}'.format(self)
        out = out + '\n' + 'sectionPathMode = {0.sectionPathMode}'.format(self)
        if self.sectionPathMode == 'section_path_subdiv':
            out = out + '\n' + 'minSpaceX = ' + str(self.minSpaceX) + ' # (-1 for automatical computation)'
            out = out + '\n' + 'minSpaceY = ' + str(self.minSpaceY) + ' # (-1 for automatical computation)'
            out = out + '\n' + 'minSpaceZ = ' + str(self.minSpaceZ) + ' # (-1 for automatical computation)'
            out = out + '\n' + 'balancedFillingFlag = {0.balancedFillingFlag}'.format(self)
        elif self.sectionPathMode == 'section_path_manual':
            out = out + '\n' + 'nsection = ' + str(self.nsection)
            out = out + '\n' + 'sectionType = ' + str(self.sectionType)
            out = out + '\n' + 'sectionLoc = ' + str(self.sectionLoc)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class DeesseXInputSection(object):
    """
    Defines input parameters for one section type (deesseX):
        nx, ny, nz: (ints) number of cells in each direction in the entire
                        simulation grid (SG);
                        should be consistent with the "parent" DeesseXInput class
                        (as defined in the "parent" DeesseXInput class)

        nv:         (int) number of variable(s) / attribute(s);
                        should be consistent with the "parent" DeesseXInput class
                        (as defined in the "parent" DeesseXInput class)

        distanceType:
                    (list (or 1-dimensional array) of ints or strings of size nv)
                        distance type (between pattern) for each variable
                        (as defined in the "parent" DeesseXInput class)

        sectionType:
                    (string or int) type of section, possible values:
                        - 'xy' or 'XY' or 0: 2D section parallel to the plane xy
                        - 'xz' or 'XZ' or 1: 2D section parallel to the plane xz
                        - 'yz' or 'YZ' or 2: 2D section parallel to the plane yz
                        - 'z' or 'Z' or 3:   1D section parallel to the axis z
                        - 'y' or 'Y' or 4:   1D section parallel to the axis y
                        - 'x' or 'X' or 5:   1D section parallel to the axis x

        nTI:        (int)
                        as in DeesseInput class (see this class)

        TI:         (1-dimensional array of Img (class))
                        as in DeesseInput class (see this class)

        simGridAsTiFlag:
                    (1-dimensional array of 'bool')
                        as in DeesseInput class (see this class)

        pdfTI:      ((nTI, nz, ny, nx) array of floats)
                        as in DeesseInput class (see this class);

        homothetyUsage:
                    (int)
                        as in DeesseInput class (see this class)

        homothetyXLocal, homothetyYLocal, homothetyZLocal:
                    (bool)
                        as in DeesseInput class (see this class)

        homothetyXRatio, homothetyYRatio, homothetyZRatio:
                    (nd array or None)
                        as in DeesseInput class (see this class)
                        note: if given "locally", the dimension of the
                        entire simulation grid is considered

        rotationUsage:
                    (int)
                        as in DeesseInput class (see this class)

        rotationAzimuthLocal, rotationDipLocal, rotationPlungeLocal:
                    (bool)
                        as in DeesseInput class (see this class)

        rotationAzimuth, rotationDip, rotationPlunge:
                    (nd array or None)
                        as in DeesseInput class (see this class)
                        note: if given "locally", the dimension of the
                        entire simulation grid is considered

        searchNeighborhoodParameters:
                    (1-dimensional array of SearchNeighborhoodParameters (class)
                        of size nv) as in DeesseInput class (see this class);

        nneighboringNode:
                    (1-dimensional array of ints of size nv)
                        as in DeesseInput class (see this class);

        maxPropInequalityNode:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class);

        neighboringNodeDensity:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class);

        simType:    (string)
                        as in DeesseInput class (see this class);
                        note: defining the type of simulation within the section

        simPathType:
                    (string)
                        as in DeesseInput class (see this class);
                        note: defining the type of path within the section

        simPathStrength:
                    (double)
                        as in DeesseInput class (see this class);
                        note: defining the type of path within the section

        simPathPower:
                    (double)
                        as in DeesseInput class (see this class);
                        note: defining the type of path within the section

        simPathUnilateralOrder:
                    (1-dimesional array of ints)
                        as in DeesseInput class (see this class);
                        note: defining the type of path within the section

        distanceThreshold:
                    (1-dimensional array of floats of size nv)
                        as in DeesseInput class (see this class);

        softProbability:
                    (1-dimensional array of SoftProbability (class) of size nv)
                        as in DeesseInput class (see this class);

        maxScanFraction:
                    (1-dimensional array of doubles of size nTI)
                        as in DeesseInput class (see this class);

        pyramidGeneralParameters:
                    (PyramidGeneralParameters (class))
                        as in DeesseInput class (see this class);
                        note: defining the general pyramid parameters
                        for the simulation within the section

        pyramidParameters:
                    (1-dimensional array of PyramidParameters (class) of size nv)
                        as in DeesseInput class (see this class);
                        note: defining the pyramid parameters
                        for the simulation within the section

        tolerance:  (float)
                        as in DeesseInput class (see this class);

        npostProcessingPathMax:
                    (int)
                        as in DeesseInput class (see this class);

        postProcessingNneighboringNode:
                    (1-dimensional array of ints of size nv)
                        as in DeesseInput class (see this class);

        postProcessingNeighboringNodeDensity:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class);

        postProcessingDistanceThreshold:
                    (1-dimensional array of floats of size nv)
                        as in DeesseInput class (see this class);

        postProcessingMaxScanFraction:
                    (1-dimensional array of doubles of size nTI)
                        as in DeesseInput class (see this class);

        postProcessingTolerance:
                    (float)
                        as in DeesseInput class (see this class);
    """

    def __init__(self,
                 nx=0, ny=0, nz=0,
                 nv=0, distanceType=None,
                 sectionType=None,
                 nTI=None, TI=None, simGridAsTiFlag=None, pdfTI=None,
                 homothetyUsage=0,
                 homothetyXLocal=False, homothetyXRatio=None,
                 homothetyYLocal=False, homothetyYRatio=None,
                 homothetyZLocal=False, homothetyZRatio=None,
                 rotationUsage=0,
                 rotationAzimuthLocal=False, rotationAzimuth=None,
                 rotationDipLocal=False,     rotationDip=None,
                 rotationPlungeLocal=False,  rotationPlunge=None,
                 searchNeighborhoodParameters=None,
                 nneighboringNode=None,
                 maxPropInequalityNode=None, neighboringNodeDensity=None,
                 simType='sim_one_by_one',
                 simPathType='random',
                 simPathStrength=None,
                 simPathPower=None,
                 simPathUnilateralOrder=None,
                 distanceThreshold=None,
                 softProbability=None,
                 maxScanFraction=None,
                 pyramidGeneralParameters=None,
                 pyramidParameters=None,
                 tolerance=0.0,
                 npostProcessingPathMax=0,
                 postProcessingNneighboringNode=None,
                 postProcessingNeighboringNodeDensity=None,
                 postProcessingDistanceThreshold=None,
                 postProcessingMaxScanFraction=None,
                 postProcessingTolerance=0.0):

        fname = 'DeesseXInputSection'

        self.ok = False # flag to "validate" the class [temporary to False]

        # grid dimension and number of variable(s)
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.nv = int(nv)

        # distance type
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
                            print(f'ERROR ({fname}): field "distanceType"...')
                            return None
                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "distanceType"...')
                return None

        # dimension
        dim = int(nx>1) + int(ny>1) + int(nz>1)

        # section type
        if sectionType is None:
            print(f'ERROR ({fname}): field "sectionType"...')
            return None

        if isinstance(sectionType, str):
            if sectionType == 'xy' or sectionType == 'XY':
                self.sectionType = 0
            elif sectionType == 'xz' or sectionType == 'XZ':
                self.sectionType = 1
            elif sectionType == 'yz' or sectionType == 'YZ':
                self.sectionType = 2
            elif sectionType == 'z' or sectionType == 'Z':
                self.sectionType = 3
            elif sectionType == 'y' or sectionType == 'Y':
                self.sectionType = 4
            elif sectionType == 'x' or sectionType == 'X':
                self.sectionType = 5
            else:
                print(f'ERROR ({fname}): field "sectionType"...')
                return None

        elif isinstance(sectionType, int):
            self.sectionType = sectionType

        else:
            print(f'ERROR ({fname}): field "sectionType"...')
            return None

        # TI, simGridAsTiFlag, nTI
        if TI is None and simGridAsTiFlag is None:
            print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag" (both None)...')
            return None

        if TI is not None:
            self.TI = np.asarray(TI).reshape(-1)

        if simGridAsTiFlag is not None:
            self.simGridAsTiFlag = np.asarray(simGridAsTiFlag, dtype='bool').reshape(-1)

        if TI is None:
            self.TI = np.array([None for i in range(len(self.simGridAsTiFlag))], dtype=object)

        if simGridAsTiFlag is None:
            self.simGridAsTiFlag = np.array([False for i in range(len(self.TI))], dtype='bool') # set dtype='bool' in case of len(self.TI)=0

        if len(self.TI) != len(self.simGridAsTiFlag):
            print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag" (not same length)...')
            return None

        for f, t in zip(self.simGridAsTiFlag, self.TI):
            if (not f and t is None) or (f and t is not None):
                print(f'ERROR ({fname}): invalid "TI / simGridAsTiFlag"...')
                return None

        if nTI is not None and nTI != len(self.TI):
            print(f'ERROR ({fname}): invalid "nTI"...')
            return None

        nTI = len(self.TI)
        self.nTI = nTI

        # pdfTI
        if nTI <= 1:
            self.pdfTI = None
        else:
            if pdfTI is None:
                p = 1./nTI
                self.pdfTI = np.repeat(p, nTI*nx*ny*nz).reshape(nTI, nz, ny, nx)
            else:
                try:
                    self.pdfTI = np.asarray(pdfTI, dtype=float).reshape(nTI, nz, ny, nx)
                except:
                    print(f'ERROR ({fname}): field "pdfTI"...')
                    return None

        # homothety
        if homothetyUsage == 1:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None

        elif homothetyUsage == 2:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None
            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyXRatio"...')
                        return None

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None
            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyYRatio"...')
                        return None

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None
            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "homothetyZRatio"...')
                        return None

        elif homothetyUsage == 0:
            self.homothetyXRatio = None
            self.homothetyYRatio = None
            self.homothetyZRatio = None

        else:
            print(f'ERROR ({fname}): invalid homothetyUsage')
            return None

        self.homothetyUsage = homothetyUsage
        self.homothetyXLocal = homothetyXLocal
        self.homothetyYLocal = homothetyYLocal
        self.homothetyZLocal = homothetyZLocal

        # rotation
        if rotationUsage == 1:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(1)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None

        elif rotationUsage == 2:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None
            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0., 0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationAzimuth"...')
                        return None

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None
            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0., 0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationDip"...')
                        return None

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None
            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0., 0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2)
                    except:
                        print(f'ERROR ({fname}): field "rotationPlunge"...')
                        return None

        elif rotationUsage == 0:
            self.rotationAzimuth = None
            self.rotationDip = None
            self.rotationPlunge = None

        else:
            print(f'ERROR ({fname}): invalid rotationUsage')
            return None

        self.rotationUsage = rotationUsage
        self.rotationAzimuthLocal = rotationAzimuthLocal
        self.rotationDipLocal = rotationDipLocal
        self.rotationPlungeLocal = rotationPlungeLocal

        # search neighborhood, number of neighbors, ...
        if searchNeighborhoodParameters is None:
            self.searchNeighborhoodParameters = np.array([SearchNeighborhoodParameters() for i in range(nv)])
        else:
            try:
                self.searchNeighborhoodParameters = np.asarray(searchNeighborhoodParameters).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "searchNeighborhoodParameters"...')
                return None

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
                print(f'ERROR ({fname}): field "nneighboringNode"...')
                return None

        if maxPropInequalityNode is None:
            self.maxPropInequalityNode = np.array([0.25 for i in range(nv)])
        else:
            try:
                self.maxPropInequalityNode = np.asarray(maxPropInequalityNode).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "maxPropInequalityNode"...')
                return None

        if neighboringNodeDensity is None:
            self.neighboringNodeDensity = np.array([1. for i in range(nv)])
        else:
            try:
                self.neighboringNodeDensity = np.asarray(neighboringNodeDensity, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "neighboringNodeDensity"...')
                return None

        # simulation type and simulation path type
        if simType not in ('sim_one_by_one', 'sim_variable_vector'):
            print(f'ERROR ({fname}): field "simType"...')
            return None

        self.simType = simType

        if simPathType not in (
                'random',
                'random_hd_distance_pdf', 'random_hd_distance_sort',
                'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort',
                'unilateral'):
            print(f'ERROR ({fname}): field "simPathType"...')
            return None

        self.simPathType = simPathType

        if simPathStrength is None:
            simPathStrength = 0.5
        if simPathPower is None:
            simPathPower = 2.0

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
                    print(f'ERROR ({fname}): field "simPathUnilateralOrder"...')
                    return None
        else:
            self.simPathUnilateralOrder = None

        # distance threshold
        if distanceThreshold is None:
            self.distanceThreshold = np.array([0.05 for i in range(nv)])
        else:
            try:
                self.distanceThreshold = np.asarray(distanceThreshold, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "distanceThreshold"...')
                return None

        # soft probability
        if softProbability is None:
            self.softProbability = np.array([SoftProbability(probabilityConstraintUsage=0) for i in range(nv)])
        else:
            try:
                self.softProbability = np.asarray(softProbability).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "softProbability"...')
                return None

        # maximal scan fraction
        if maxScanFraction is None:
            if dim == 3: # 3D simulation
                nf = 10000
            else:
                nf = 5000

            self.maxScanFraction = np.array([min(max(nf/self.TI[i].nxyz(), deesse.MPDS_MIN_MAXSCANFRACTION), deesse.MPDS_MAX_MAXSCANFRACTION) for i in range(nTI)])
        else:
            try:
                self.maxScanFraction = np.asarray(maxScanFraction).reshape(nTI)
            except:
                print(f'ERROR ({fname}): field "maxScanFraction"...')
                return None

        # pyramids
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
                print(f'ERROR ({fname}): field "pyramidParameters"...')
                return None

        # tolerance and post-processing
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
                print(f'ERROR ({fname}): field "postProcessingNneighboringNode"...')
                return None

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
                print(f'ERROR ({fname}): field "postProcessingNeighboringNodeDensity"...')
                return None

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
                print(f'ERROR ({fname}): field "postProcessingDistanceThreshold"...')
                return None

        if postProcessingMaxScanFraction is None:
            self.postProcessingMaxScanFraction = np.array([min(deesse.MPDS_POST_PROCESSING_MAX_SCAN_FRACTION_DEFAULT, self.maxScanFraction[i]) for i in range(nTI)], dtype=float)

        else:
            try:
                self.postProcessingMaxScanFraction = np.asarray(postProcessingMaxScanFraction, dtype=float).reshape(nTI)
            except:
                print(f'ERROR ({fname}): field "postProcessingMaxScanFraction"...')
                return None

        self.postProcessingTolerance = postProcessingTolerance

        self.ok = True # flag to "validate" the class

    # ------------------------------------------------------------------------
    # def __str__(self):
    def __repr__(self):
        out = '*** DeesseXInputSection object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class DeesseXInput(object):
    """
    Defines general input parameters for deesseX (cross-simulation/X-simulation):
        simName:    (str) simulation name (not useful)
        nx, ny, nz: (ints) number of simulation grid (SG) cells in each direction
                        as in DeesseInput class (see this class)
        sx, sy, sz: (floats) cell size in each direction
                        as in DeesseInput class (see this class)
        ox, oy, oz: (floats) origin of the SG (bottom-lower-left corner)
                        as in DeesseInput class (see this class)
        nv:         (int) number of variable(s) / attribute(s), should be
                        at least 1
                        as in DeesseInput class (see this class)

        varname:    (list of strings of length nv) variable names
                        as in DeesseInput class (see this class)

        outputVarFlag:
                    (1-dimensional array of 'bool', of size nv)
                        as in DeesseInput class (see this class)

        outputSectionTypeFlag:
                    (bool) indicates if "section type" map(s) is (are) retrieved
                        in output; one file per realization if section path mode
                        is set to 'section_path_random', and one file in all
                        otherwise (same for each realization), (see
                        sectionPathMode in DeesseXInputSectionPath class);
                        "section type" is an index identifiying the type of
                        section (see sectionType in DeesseXInputSectionPath class)

        outputSectionStepFlag:
                    (bool) indicates if "section step" map(s) is (are) retrieved
                        in output; one file per realization if section path mode
                        is set to 'section_path_random', and one file in all
                        otherwise (same for each realization (see sectionPathMode
                        in DeesseXInputSectionPath class); "section step" is the
                        step index of simulation by deesse of (a bunch of)
                        sections of same type


        outputReportFlag:
                    (bool)
                        as in DeesseInput class (see this class)

        outputReportFileName:
                    (string)
                        as in DeesseInput class (see this class)

        dataImage:  (1-dimensional array of Img (class), or None)
                        as in DeesseInput class (see this class)
        dataPointSet:
                    (1-dimensional array of PointSet (class), or None)
                        as in DeesseInput class (see this class)

        mask:       ((nz, ny, nx) array of ints, or None)
                        as in DeesseInput class (see this class)

        expMax:     (float)
                        as in DeesseInput class (see this class)

        normalizingType:
                    (string)
                        as in DeesseInput class (see this class)

        rescalingMode:
                    (list of strings of length nv)
                        as in DeesseInput class (see this class)

        rescalingTargetMin:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        rescalingTargetMax:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        rescalingTargetMean:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        rescalingTargetLength:
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        relativeDistanceFlag:
                    (1-dimensional array of 'bool', of size nv)
                        as in DeesseInput class (see this class)
        distanceType:
                    (list (or 1-dimensional array) of ints or strings of size nv)
                        as in DeesseInput class (see this class)

        powerLpDistance
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        powerLpDistanceInv
                    (1-dimensional array of doubles of size nv)
                        as in DeesseInput class (see this class)

        conditioningWeightFactor:
                    (1-dimensional array of floats of size nv)
                        as in DeesseInput class (see this class)

        sectionPath_parameters:
                    (DeesseXInputSectionPath (class))
                       defines the overall strategy of simulation
                       as a succession of section (see this class)

        section_parameters:
                    (1-dimensional array of DeesseXInputSection (class))
                        each component defines the parameter for
                        one section type (see this class);

        seed:       (int) initial seed
                        as in DeesseInput class (see this class)

        seedIncrement:
                    (int) increment seed
                        as in DeesseInput class (see this class)

        nrealization:
                    (int) number of realization(s)
                        as in DeesseInput class (see this class)

    Note: in output simulated images (obtained by running deesseX), the names
        of the output variables are set to <vname>_real<n>, where
            - <vname> is the name of the variable,
            - <n> is the realization index (starting from 0)
            [<n> is written on 5 digits, with leading zeros]
    """

    def __init__(self,
                 simName='deesseX_py',
                 nx=0,   ny=0,   nz=0,
                 sx=1.0, sy=1.0, sz=1.0,
                 ox=0.0, oy=0.0, oz=0.0,
                 nv=0, varname=None, outputVarFlag=None,
                 outputSectionTypeFlag=False, #outputSectionTypeFileName=None,
                 outputSectionStepFlag=False, #outputSectionStepFileName=None,
                 outputReportFlag=False, outputReportFileName=None,
                 dataImage=None, dataPointSet=None,
                 mask=None,
                 expMax=0.05,
                 normalizingType='linear',
                 rescalingMode=None,
                 rescalingTargetMin=None, rescalingTargetMax=None,
                 rescalingTargetMean=None, rescalingTargetLength=None,
                 relativeDistanceFlag=None,
                 distanceType=None,
                 powerLpDistance=None,
                 conditioningWeightFactor=None,
                 sectionPath_parameters=None,
                 section_parameters=None,
                 seed=1234,
                 seedIncrement=1,
                 nrealization=1):

        fname = 'DeesseXInput'

        self.ok = False # flag to "validate" the class [temporary to False]

        # consoleAppFlag
        self.consoleAppFlag = False

        # simulation name
        self.simName = simName

        # grid definition and variable(s)
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)
        self.ox = float(ox)
        self.oy = float(oy)
        self.oz = float(oz)
        self.nv = int(nv)
        if varname is None:
            self.varname = ["V{:d}".format(i) for i in range(nv)]
        else:
            try:
                self.varname = list(np.asarray(varname).reshape(nv))
            except:
                print(f'ERROR ({fname}): field "varname"...')
                return None

        # outputVarFlag
        if outputVarFlag is None:
            self.outputVarFlag = np.array([True for i in range(nv)], dtype='bool')
        else:
            try:
                self.outputVarFlag = np.asarray(outputVarFlag, dtype='bool').reshape(nv)
            except:
                print(f'ERROR ({fname}): field "outputVarFlag"...')
                return None

        # output maps
        self.outputSectionTypeFlag = outputSectionTypeFlag
        # self.outputSectionTypeFileName = None # no output file!

        self.outputSectionStepFlag = outputSectionStepFlag
        # self.outputSectionStepFileName = None # no output file!

        # report
        self.outputReportFlag = outputReportFlag
        if outputReportFileName is None:
            self.outputReportFileName ='dsX.log'
        else:
            self.outputReportFileName = outputReportFileName

        # conditioning data image
        if dataImage is None:
            self.dataImage = None
        else:
            self.dataImage = np.asarray(dataImage).reshape(-1)

        # conditioning point set
        if dataPointSet is None:
            self.dataPointSet = None
        else:
            self.dataPointSet = np.asarray(dataPointSet).reshape(-1)

        # mask
        if mask is None:
            self.mask = None
        else:
            try:
                self.mask = np.asarray(mask).reshape(nz, ny, nx)
            except:
                print(f'ERROR ({fname}): field "mask"...')
                return None

        # expMax
        self.expMax = expMax

        # normalizing type
        # if normalizingType not in ('linear', 'uniform', 'normal'):
        #     print('ERRROR: (DeesseXInput) field "normalizingType"')
        #     return None

        self.normalizingType = normalizingType

        # rescaling
        if rescalingMode is None:
            self.rescalingMode = ['none' for i in range(nv)]
        else:
            try:
                self.rescalingMode = list(np.asarray(rescalingMode).reshape(nv))
            except:
                print(f'ERROR ({fname}): field "rescalingMode"...')
                return None

        if rescalingTargetMin is None:
            self.rescalingTargetMin = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMin = np.asarray(rescalingTargetMin, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMin"...')
                return None

        if rescalingTargetMax is None:
            self.rescalingTargetMax = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMax = np.asarray(rescalingTargetMax, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMax"...')
                return None

        if rescalingTargetMean is None:
            self.rescalingTargetMean = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMean = np.asarray(rescalingTargetMean, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetMean"...')
                return None

        if rescalingTargetLength is None:
            self.rescalingTargetLength = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetLength = np.asarray(rescalingTargetLength, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "rescalingTargetLength"...')
                return None

        # distance, ...
        if relativeDistanceFlag is None:
            self.relativeDistanceFlag = np.array([False for i in range(nv)], dtype='bool') # set dtype='bool' in case of nv=0
        else:
            try:
                self.relativeDistanceFlag = np.asarray(relativeDistanceFlag, dtype='bool').reshape(nv)
            except:
                print(f'ERROR ({fname}): field "relativeDistanceFlag"...')
                return None

        if powerLpDistance is None:
            self.powerLpDistance = np.array([1. for i in range(nv)])
        else:
            try:
                self.powerLpDistance = np.asarray(powerLpDistance, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "powerLpDistance"...')
                return None

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
                            print(f'ERROR ({fname}): field "distanceType"...')
                            return None
                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "distanceType"...')
                return None

        # conditioning weight
        if conditioningWeightFactor is None:
            self.conditioningWeightFactor = np.array([1. for i in range(nv)])
        else:
            try:
                self.conditioningWeightFactor = np.asarray(conditioningWeightFactor, dtype=float).reshape(nv)
            except:
                print(f'ERROR ({fname}): field "conditioningWeightFactor"...')
                return None

        # sectionPath_parameters
        if sectionPath_parameters is None:
            print(f'ERROR ({fname}): field "sectionPath_parameters" (must be specified)...')
            return None

        self.sectionPath_parameters = sectionPath_parameters

        # section_parameters
        if section_parameters is None:
            print(f'ERROR ({fname}): field "section_parameters" (must be specified)...')
            return None

        self.section_parameters = np.asarray(section_parameters).reshape(-1)

        # seed, ...
        if seed is None:
            seed = np.random.randint(1,1000000)
        self.seed = seed
        self.seedIncrement = seedIncrement

        # number of realization(s)
        self.nrealization = nrealization

        self.ok = True # flag to "validate" the class

    # ------------------------------------------------------------------------
    # def __str__(self):
    def __repr__(self):
        out = '*** DeesseInput object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ----------------------------------------------------------------------------
def deesseX_input_sectionPath_py2C(sectionPath_parameters):
    """
    Converts section path parameters (deesseX) from python to C
    (MPDS_XSECTIONPARAMETERS).

    :param sectionPath_parameters:
                                (DeesseXInputSectionPath class) section path
                                    parameters (strategy of simulation) (python)

    :return mpds_xsectionParameters:
                                (MPDS_XSECTIONPARAMETERS *) corresponding
                                    parameters (C struct)
    """

    fname = 'deesseX_input_sectionPath_py2C'

    mpds_xsectionParameters = deesse.malloc_MPDS_XSECTIONPARAMETERS()
    deesse.MPDSInitXSectionParameters(mpds_xsectionParameters)

    # XSectionMode
    sectionMode_dict = {
        'section_xy_xz_yz' : deesse.SECTION_XY_XZ_YZ,
        'section_xy_yz_xz' : deesse.SECTION_XY_YZ_XZ,
        'section_xz_xy_yz' : deesse.SECTION_XZ_XY_YZ,
        'section_xz_yz_xy' : deesse.SECTION_XZ_YZ_XY,
        'section_yz_xy_xz' : deesse.SECTION_YZ_XY_XZ,
        'section_yz_xz_xy' : deesse.SECTION_YZ_XZ_XY,
        'section_xy_xz'    : deesse.SECTION_XY_XZ,
        'section_xz_xy'    : deesse.SECTION_XZ_XY,
        'section_xy_yz'    : deesse.SECTION_XY_YZ,
        'section_yz_xy'    : deesse.SECTION_YZ_XY,
        'section_xz_yz'    : deesse.SECTION_XZ_YZ,
        'section_yz_xz'    : deesse.SECTION_YZ_XZ,
        'section_xy_z'     : deesse.SECTION_XY_Z,
        'section_z_xy'     : deesse.SECTION_Z_XY,
        'section_xz_y'     : deesse.SECTION_XZ_Y,
        'section_y_xz'     : deesse.SECTION_Y_XZ,
        'section_yz_x'     : deesse.SECTION_YZ_X,
        'section_x_yz'     : deesse.SECTION_X_YZ,
        'section_x_y_z'    : deesse.SECTION_X_Y_Z,
        'section_x_z_y'    : deesse.SECTION_X_Z_Y,
        'section_y_x_z'    : deesse.SECTION_Y_X_Z,
        'section_y_z_x'    : deesse.SECTION_Y_Z_X,
        'section_z_x_y'    : deesse.SECTION_Z_X_Y,
        'section_z_y_x'    : deesse.SECTION_Z_Y_X,
        'section_x_y'      : deesse.SECTION_X_Y,
        'section_y_x'      : deesse.SECTION_Y_X,
        'section_x_z'      : deesse.SECTION_X_Z,
        'section_z_x'      : deesse.SECTION_Z_X,
        'section_y_z'      : deesse.SECTION_Y_Z,
        'section_z_y'      : deesse.SECTION_Z_Y
    }
    try:
        mpds_xsectionParameters.XSectionMode = sectionMode_dict[sectionPath_parameters.sectionMode]
    except:
        print(f'ERROR ({fname}): section mode unknown')
        return None

    # XSectionPathMode and other relevant fields
    if sectionPath_parameters.sectionPathMode == 'section_path_random':
        mpds_xsectionParameters.XSectionPathMode = deesse.SECTION_PATH_RANDOM

    elif sectionPath_parameters.sectionPathMode == 'section_path_pow_2':
        mpds_xsectionParameters.XSectionPathMode = deesse.SECTION_PATH_POW_2

    elif sectionPath_parameters.sectionPathMode == 'section_path_subdiv':
        mpds_xsectionParameters.XSectionPathMode = deesse.SECTION_PATH_SUBDIV
        mpds_xsectionParameters.minSpaceX = sectionPath_parameters.minSpaceX
        mpds_xsectionParameters.minSpaceY = sectionPath_parameters.minSpaceY
        mpds_xsectionParameters.minSpaceZ = sectionPath_parameters.minSpaceZ
        if sectionPath_parameters.balancedFillingFlag:
            mpds_xsectionParameters.balancedFillingFlag = deesse.TRUE
        else:
            mpds_xsectionParameters.balancedFillingFlag = deesse.FALSE

    elif sectionPath_parameters.sectionPathMode == 'section_path_manual':
        mpds_xsectionParameters.XSectionPathMode = deesse.SECTION_PATH_MANUAL
        ns = int(sectionPath_parameters.nsection)
        mpds_xsectionParameters.nsection = ns
        if ns > 0:
            mpds_xsectionParameters.sectionType = deesse.new_int_array(ns)
            deesse.mpds_set_int_vector_from_array(
                mpds_xsectionParameters.sectionType, 0,
                np.asarray(sectionPath_parameters.sectionType, dtype='intc').reshape(ns))
            mpds_xsectionParameters.sectionLoc = deesse.new_int_array(ns)
            deesse.mpds_set_int_vector_from_array(
                mpds_xsectionParameters.sectionLoc, 0,
                np.asarray(sectionPath_parameters.sectionLoc, dtype='intc').reshape(ns))
    else:
        print(f'ERROR ({fname}): section path type unknown')
        return None

    return mpds_xsectionParameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_sectionPath_C2py(mpds_xsectionParameters):
    """
    Converts section path parameters (deesseX) from C (MPDS_XSECTIONPARAMETERS)
    to python.

    :param mpds_xsectionParameters:
                                (MPDS_XSECTIONPARAMETERS *) C parameters
                                    (C struct)

    :return sectionPath_parameters:
                                (DeesseXInputSectionPath class) section path
                                    parameters (strategy of simulation) (python)
    """

    fname = 'deesseX_input_sectionPath_C2py'

    # sectionMode
    sectionMode_dict = {
        deesse.SECTION_XY_XZ_YZ : 'section_xy_xz_yz',
        deesse.SECTION_XY_YZ_XZ : 'section_xy_yz_xz',
        deesse.SECTION_XZ_XY_YZ : 'section_xz_xy_yz',
        deesse.SECTION_XZ_YZ_XY : 'section_xz_yz_xy',
        deesse.SECTION_YZ_XY_XZ : 'section_yz_xy_xz',
        deesse.SECTION_YZ_XZ_XY : 'section_yz_xz_xy',
        deesse.SECTION_XY_XZ    : 'section_xy_xz',
        deesse.SECTION_XZ_XY    : 'section_xz_xy',
        deesse.SECTION_XY_YZ    : 'section_xy_yz',
        deesse.SECTION_YZ_XY    : 'section_yz_xy',
        deesse.SECTION_XZ_YZ    : 'section_xz_yz',
        deesse.SECTION_YZ_XZ    : 'section_yz_xz',
        deesse.SECTION_XY_Z     : 'section_xy_z',
        deesse.SECTION_Z_XY     : 'section_z_xy',
        deesse.SECTION_XZ_Y     : 'section_xz_y',
        deesse.SECTION_Y_XZ     : 'section_y_xz',
        deesse.SECTION_YZ_X     : 'section_yz_x',
        deesse.SECTION_X_YZ     : 'section_x_yz',
        deesse.SECTION_X_Y_Z    : 'section_x_y_z',
        deesse.SECTION_X_Z_Y    : 'section_x_z_y',
        deesse.SECTION_Y_X_Z    : 'section_y_x_z',
        deesse.SECTION_Y_Z_X    : 'section_y_z_x',
        deesse.SECTION_Z_X_Y    : 'section_z_x_y',
        deesse.SECTION_Z_Y_X    : 'section_z_y_x',
        deesse.SECTION_X_Y      : 'section_x_y',
        deesse.SECTION_Y_X      : 'section_y_x',
        deesse.SECTION_X_Z      : 'section_x_z',
        deesse.SECTION_Z_X      : 'section_z_x',
        deesse.SECTION_Y_Z      : 'section_y_z',
        deesse.SECTION_Z_Y      : 'section_z_y'
    }
    try:
        sectionMode = sectionMode_dict[mpds_xsectionParameters.XSectionMode]
    except:
        print(f'ERROR ({fname}): section mode unknown')
        return None

    # ... sectionPathMode other relevant fields
    # ... ... default parameters
    minSpaceX = None
    minSpaceY = None
    minSpaceZ = None
    balancedFillingFlag = True
    nsection = 0
    sectionType = None
    sectionLoc = None

    if mpds_xsectionParameters.XSectionPathMode == deesse.SECTION_PATH_RANDOM:
        sectionPathMode = 'section_path_random'

    elif mpds_xsectionParameters.XSectionPathMode == deesse.SECTION_PATH_POW_2:
        sectionPathMode = 'section_path_pow_2'

    elif mpds_xsectionParameters.XSectionPathMode == deesse.SECTION_PATH_SUBDIV:
        sectionPathMode = 'section_path_subdiv'
        minSpaceX = mpds_xsectionParameters.minSpaceX
        minSpaceY = mpds_xsectionParameters.minSpaceY
        minSpaceZ = mpds_xsectionParameters.minSpaceZ
        balancedFillingFlag = bool(int.from_bytes(mpds_xsectionParameters.balancedFillingFlag.encode('utf-8'), byteorder='big'))

    elif mpds_xsectionParameters.XSectionPathMode == deesse.SECTION_PATH_MANUAL:
        sectionPathMode = 'section_path_manual'
        nsection = mpds_xsectionParameters.nsection
        if nsection > 0:
            sectionType = np.zeros(nsection, dtype='intc')
            deesse.mpds_get_array_from_int_vector(mpds_xsectionParameters.sectionType, 0, sectionType)
            sectionType = distanceType.astype('int')

            sectionLoc = np.zeros(nsection, dtype='intc')
            deesse.mpds_get_array_from_int_vector(mpds_xsectionParameters.sectionLoc, 0, sectionLoc)
            sectionLoc = distanceType.astype('int')

    else:
        print(f'ERROR ({fname}): section path type unknown')
        return None

    sectionPath_parameters = DeesseXInputSectionPath(
        sectionMode=sectionMode,
        sectionPathMode=sectionPathMode,
        minSpaceX=minSpaceX,
        minSpaceY=minSpaceY,
        minSpaceZ=minSpaceZ,
        balancedFillingFlag=balancedFillingFlag,
        nsection=nsection,
        sectionType=sectionType,
        sectionLoc=sectionLoc)

    return sectionPath_parameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_section_py2C(
        section_parameters,
        sectionType,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        nv):
    """
    Converts section parameters (parameters for one section) (deesseX) from
    python to C (MPDS_XSUBSIMINPUT).

    :param section_parameters:  (DeesseXInputSection class) section parameters
                                    (parameters for one section) (python)
    :param sectionType:         (int) id of the section type
    :param nx, ny, nz:          (ints) number of simulation grid (SG) cells in
                                    each direction
    :param sx, sy, sz:          (floats) cell size in each direction
    :param ox, oy, oz:          (floats) origin of the SG
                                    (bottom-lower-left corner)
    :param nv:                  (int) number of variable(s) / attribute(s)

    :return mpds_xsubsiminput:  (MPDS_XSUBSIMINPUT *) corresponding parameters
                                    (C struct)
    """

    fname = 'deesseX_input_section_py2C'

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    nv = int(nv)

    nTI = int(section_parameters.nTI)

    # Allocate mpds_xsubsiminput
    mpds_xsubsiminput = deesse.malloc_MPDS_XSUBSIMINPUT()

    # Init mpds_xsubsiminput
    deesse.MPDSInitXSubSimInput(mpds_xsubsiminput)

    # mpds_xsubsiminput.sectionType
    mpds_xsubsiminput.sectionType = sectionType

    # mpds_xsubsiminput.nvar
    mpds_xsubsiminput.nvar = nv

    # mpds_xsubsiminput.ntrainImage
    mpds_xsubsiminput.ntrainImage = nTI

    # mpds_xsubsiminput.simGridAsTiFlag
    deesse.mpds_xsub_allocate_and_set_simGridAsTiFlag(mpds_xsubsiminput, np.array([int(i) for i in section_parameters.simGridAsTiFlag], dtype='bool')) # dtype='intc'))

    # mpds_xsubsiminput.trainImage
    mpds_xsubsiminput.trainImage = deesse.new_MPDS_IMAGE_array(nTI)
    for i, ti in enumerate(section_parameters.TI):
        if ti is not None:
            im_c = img_py2C(ti)
            deesse.MPDS_IMAGE_array_setitem(mpds_xsubsiminput.trainImage, i, im_c)
            # deesse.free_MPDS_IMAGE(im_c)
            #
            # deesse.MPDS_IMAGE_array_setitem(mpds_xsubsiminput.trainImage, i, img_py2C(ti))

    # mpds_xsubsiminput.pdfTrainImage
    if nTI > 1:
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=nTI, val=section_parameters.pdfTI)
        mpds_xsubsiminput.pdfTrainImage = img_py2C(im)

    # Homothety:
    #   mpds_xsubsiminput.homothetyUsage
    #   mpds_xsubsiminput.homothety[XYZ]RatioImageFlag
    #   mpds_xsubsiminput.homothety[XYZ]RatioImage
    #   mpds_xsubsiminput.homothety[XYZ]RatioValue
    mpds_xsubsiminput.homothetyUsage = section_parameters.homothetyUsage
    if section_parameters.homothetyUsage == 1:
        if section_parameters.homothetyXLocal:
            mpds_xsubsiminput.homothetyXRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.homothetyXRatio)
            mpds_xsubsiminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyXRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyXRatioValue, 0,
                np.asarray(section_parameters.homothetyXRatio).reshape(1))

        if section_parameters.homothetyYLocal:
            mpds_xsubsiminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.homothetyYRatio)
            mpds_xsubsiminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyYRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyYRatioValue, 0,
                np.asarray(section_parameters.homothetyYRatio).reshape(1))

        if section_parameters.homothetyZLocal:
            mpds_xsubsiminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.homothetyZRatio)
            mpds_xsubsiminput.homothetyZRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyZRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyZRatioValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyZRatioValue, 0,
                np.asarray(section_parameters.homothetyZRatio).reshape(1))

    elif section_parameters.homothetyUsage == 2:
        if section_parameters.homothetyXLocal:
            mpds_xsubsiminput.homothetyXRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.homothetyXRatio)
            mpds_xsubsiminput.homothetyXRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyXRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyXRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyXRatioValue, 0,
                np.asarray(section_parameters.homothetyXRatio).reshape(2))

        if section_parameters.homothetyYLocal:
            mpds_xsubsiminput.homothetyYRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.homothetyYRatio)
            mpds_xsubsiminput.homothetyYRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyYRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyYRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyYRatioValue, 0,
                np.asarray(section_parameters.homothetyYRatio).reshape(2))

        if section_parameters.homothetyZLocal:
            mpds_xsubsiminput.homothetyZRatioImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.homothetyZRatio)
            mpds_xsubsiminput.homothetyZRatioImage = img_py2C(im)

        else:
            mpds_xsubsiminput.homothetyZRatioImageFlag = deesse.FALSE
            mpds_xsubsiminput.homothetyZRatioValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.homothetyZRatioValue, 0,
                np.asarray(section_parameters.homothetyZRatio).reshape(2))

    # Rotation:
    #   mpds_xsubsiminput.rotationUsage
    #   mpds_xsubsiminput.rotation[Azimuth|Dip|Plunge]ImageFlag
    #   mpds_xsubsiminput.rotation[Azimuth|Dip|Plunge]Image
    #   mpds_xsubsiminput.rotation[Azimuth|Dip|Plunge]Value
    mpds_xsubsiminput.rotationUsage = section_parameters.rotationUsage
    if section_parameters.rotationUsage == 1:
        if section_parameters.rotationAzimuthLocal:
            mpds_xsubsiminput.rotationAzimuthImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.rotationAzimuth)
            mpds_xsubsiminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationAzimuthValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationAzimuthValue, 0,
                np.asarray(section_parameters.rotationAzimuth).reshape(1))

        if section_parameters.rotationDipLocal:
            mpds_xsubsiminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.rotationDip)
            mpds_xsubsiminput.rotationDipImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationDipImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationDipValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationDipValue, 0,
                np.asarray(section_parameters.rotationDip).reshape(1))

        if section_parameters.rotationPlungeLocal:
            mpds_xsubsiminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=section_parameters.rotationPlunge)
            mpds_xsubsiminput.rotationPlungeImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationPlungeImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationPlungeValue = deesse.new_real_array(1)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationPlungeValue, 0,
                np.asarray(section_parameters.rotationPlunge).reshape(1))

    elif section_parameters.rotationUsage == 2:
        if section_parameters.rotationAzimuthLocal:
            mpds_xsubsiminput.rotationAzimuthImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.rotationAzimuth)
            mpds_xsubsiminput.rotationAzimuthImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationAzimuthImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationAzimuthValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationAzimuthValue, 0,
                np.asarray(section_parameters.rotationAzimuth).reshape(2))

        if section_parameters.rotationDipLocal:
            mpds_xsubsiminput.rotationDipImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.rotationDip)
            mpds_xsubsiminput.rotationDipImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationDipImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationDipValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationDipValue, 0,
                np.asarray(section_parameters.rotationDip).reshape(2))

        if section_parameters.rotationPlungeLocal:
            mpds_xsubsiminput.rotationPlungeImageFlag = deesse.TRUE
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=2, val=section_parameters.rotationPlunge)
            mpds_xsubsiminput.rotationPlungeImage = img_py2C(im)

        else:
            mpds_xsubsiminput.rotationPlungeImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationPlungeValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationPlungeValue, 0,
                np.asarray(section_parameters.rotationPlunge).reshape(2))

    # mpds_xsubsiminput.searchNeighborhoodParameters
    mpds_xsubsiminput.searchNeighborhoodParameters = deesse.new_MPDS_SEARCHNEIGHBORHOODPARAMETERS_array(nv)
    for i, sn in enumerate(section_parameters.searchNeighborhoodParameters):
        sn_c = search_neighborhood_parameters_py2C(sn)
        if sn_c is None:
            print(f'ERROR ({fname}): can not convert search neighborhood parameters from python to C')
            return None
        deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_setitem(
            mpds_xsubsiminput.searchNeighborhoodParameters, i, sn_c)
        # deesse.free_MPDS_SEARCHNEIGHBORHOODPARAMETERS(sn_c)

    # mpds_xsubsiminput.nneighboringNode
    mpds_xsubsiminput.nneighboringNode = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_xsubsiminput.nneighboringNode, 0,
        np.asarray(section_parameters.nneighboringNode, dtype='intc').reshape(nv))

    # mpds_xsubsiminput.maxPropInequalityNode
    mpds_xsubsiminput.maxPropInequalityNode = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.maxPropInequalityNode, 0,
        np.asarray(section_parameters.maxPropInequalityNode).reshape(nv))

    # mpds_xsubsiminput.neighboringNodeDensity
    mpds_xsubsiminput.neighboringNodeDensity = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.neighboringNodeDensity, 0,
        np.asarray(section_parameters.neighboringNodeDensity).reshape(nv))

    # mpds_xsubsiminput.simAndPathParameters
    mpds_xsubsiminput.simAndPathParameters = set_simAndPathParameters_C(
        section_parameters.simType,
        section_parameters.simPathType,
        section_parameters.simPathStrength,
        section_parameters.simPathPower,
        section_parameters.simPathUnilateralOrder)
    if mpds_xsubsiminput.simAndPathParameters is None:
        print(f'ERROR ({fname}): can not set "simAndPathParameters" in C')
        return None

    # mpds_xsubsiminput.distanceThreshold
    mpds_xsubsiminput.distanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsubsiminput.distanceThreshold, 0,
        np.asarray(section_parameters.distanceThreshold).reshape(nv))

    # mpds_xsubsiminput.softProbability ...
    mpds_xsubsiminput.softProbability = deesse.new_MPDS_SOFTPROBABILITY_array(nv)

    # ... for each variable ...
    for i, sp in enumerate(section_parameters.softProbability):
        sp_c = softProbability_py2C(sp,
                                    nx, ny, nz,
                                    sx, sy, sz,
                                    ox, oy, oz)
        if sp_c is None:
            print(f'ERROR ({fname}): can not set soft probability parameters in C')
            return None
        deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_xsubsiminput.softProbability, i, sp_c)
        # deesse.free_MPDS_SOFTPROBABILITY(sp_c)

    # mpds_xsubsiminput.maxScanFraction
    mpds_xsubsiminput.maxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.maxScanFraction, 0,
            np.asarray(section_parameters.maxScanFraction).reshape(nTI))

    # mpds_xsubsiminput.pyramidGeneralParameters ...
    mpds_xsubsiminput.pyramidGeneralParameters = pyramidGeneralParameters_py2C(section_parameters.pyramidGeneralParameters)
    if mpds_xsubsiminput.pyramidGeneralParameters is None:
        print(f'ERROR ({fname}): can not set pyramid general parameters in C')
        return None

    # mpds_xsubsiminput.pyramidParameters ...
    mpds_xsubsiminput.pyramidParameters = deesse.new_MPDS_PYRAMIDPARAMETERS_array(nv)

    # ... for each variable ...
    for i, pp in enumerate(section_parameters.pyramidParameters):
        pp_c = pyramidParameters_py2C(pp)
        if pp_c is None:
            print(f'ERROR ({fname}): can not set pyramid parameters in C')
            return None

        deesse.MPDS_PYRAMIDPARAMETERS_array_setitem(mpds_xsubsiminput.pyramidParameters, i, pp_c)
        # deesse.free_MPDS_PYRAMIDPARAMETERS(pp_c)

    # mpds_xsubsiminput.tolerance
    mpds_xsubsiminput.tolerance = section_parameters.tolerance

    # mpds_xsubsiminput.npostProcessingPathMax
    mpds_xsubsiminput.npostProcessingPathMax = section_parameters.npostProcessingPathMax

    # mpds_xsubsiminput.postProcessingNneighboringNode
    mpds_xsubsiminput.postProcessingNneighboringNode = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_xsubsiminput.postProcessingNneighboringNode, 0,
            np.asarray(section_parameters.postProcessingNneighboringNode, dtype='intc').reshape(nv))

    # mpds_xsubsiminput.postProcessingNeighboringNodeDensity
    mpds_xsubsiminput.postProcessingNeighboringNodeDensity = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.postProcessingNeighboringNodeDensity, 0,
            np.asarray(section_parameters.postProcessingNeighboringNodeDensity).reshape(nv))

    # mpds_xsubsiminput.postProcessingDistanceThreshold
    mpds_xsubsiminput.postProcessingDistanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsubsiminput.postProcessingDistanceThreshold, 0,
            np.asarray(section_parameters.postProcessingDistanceThreshold).reshape(nv))

    # mpds_xsubsiminput.postProcessingMaxScanFraction
    mpds_xsubsiminput.postProcessingMaxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.postProcessingMaxScanFraction, 0,
            np.asarray(section_parameters.postProcessingMaxScanFraction).reshape(nTI))

    # mpds_xsubsiminput.postProcessingTolerance
    mpds_xsubsiminput.postProcessingTolerance = section_parameters.postProcessingTolerance

    return mpds_xsubsiminput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_section_C2py(
        mpds_xsubsiminput,
        sectionType,
        nx, ny, nz,
        nv, distanceType):
    """
    Converts section parameters (parameters for one section) (deesseX) from C
    (MPDS_XSUBSIMINPUT) to python.

    :param mpds_xsubsiminput:   (MPDS_XSUBSIMINPUT *) C parameters
                                    (C struct)
    :param sectionType:         (int) id of the section type
    :param nx, ny, nz:          (ints) number of simulation grid (SG) cells in
                                    each direction
    :param nv:                  (int) number of variable(s) / attribute(s)
    :param distanceType:        (list (or 1-dimensional array) of ints or
                                    strings of size nv) distance type (between
                                    pattern) for each variable

    :return section_parameters: (DeesseXInputSection class) section parameters
                                    (parameters for one section) (python)
    """

    fname = 'deesseX_input_section_C2py'

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    nv = int(nv)

    # # sectionType
    # sectionType = mpds_xsubsiminput.sectionType

    # TI, simGridAsTiFlag, nTI
    nTI = mpds_xsubsiminput.ntrainImage
    simGridAsTiFlag = np.zeros(nTI, dtype='intc')
    deesse.mpds_xsub_get_simGridAsTiFlag(mpds_xsubsiminput, simGridAsTiFlag)
    simGridAsTiFlag = simGridAsTiFlag.astype('bool')
    TI = np.array(nTI*[None])
    for i in range(nTI):
        if not simGridAsTiFlag[i]:
            im = deesse.MPDS_IMAGE_array_getitem(mpds_xsubsiminput.trainImage, i)
            TI[i] = img_C2py(im)

    # pdfTI
    pdfTI = None
    if nTI > 1:
        im = img_C2py(mpds_xsubsiminput.pdfTrainImage)
        pdfTI = im.val

    # homothety
    homothetyUsage = mpds_xsubsiminput.homothetyUsage
    homothetyXLocal = False
    homothetyXRatio = None
    homothetyYLocal = False
    homothetyYRatio = None
    homothetyZLocal = False
    homothetyZRatio = None
    if homothetyUsage == 1:
        homothetyXLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyXRatioImage)
            homothetyXRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyXRatioValue, 0, v)
            homothetyXRatio = v[0]

        homothetyYLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyYRatioImage)
            homothetyYRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyYRatioValue, 0, v)
            homothetyYRatio = v[0]

        homothetyZLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyZRatioImage)
            homothetyZRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyZRatioValue, 0, v)
            homothetyZRatio = v[0]

    elif homothetyUsage == 2:
        homothetyXLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyXRatioImage)
            homothetyXRatio = im.val
        else:
            homothetyXRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyXRatioValue, 0, homothetyXRatio)

        homothetyYLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyYRatioImage)
            homothetyYRatio = im.val
        else:
            homothetyYRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyYRatioValue, 0, homothetyYRatio)

        homothetyZLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyZRatioImage)
            homothetyZRatio = im.val
        else:
            homothetyZRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyZRatioValue, 0, homothetyZRatio)

    # rotation
    rotationUsage = mpds_xsubsiminput.rotationUsage
    rotationAzimuthLocal = False
    rotationAzimuth = None
    rotationDipLocal = False
    rotationDip = None
    rotationPlungeLocal = False
    rotationPlunge = None
    if rotationUsage == 1:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_xsubsiminput.rotationAzimuthImage)
            rotationAzimuth = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationAzimuthValue, 0, v)
            rotationAzimuth = v[0]

        rotationDipLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_xsubsiminput.rotationDipImage)
            rotationDip = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationDipValue, 0, v)
            rotationDip = v[0]

        rotationPlungeLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_xsubsiminput.rotationPlungeImage)
            rotationPlunge = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationPlungeValue, 0, v)
            rotationPlunge = v[0]

    elif rotationUsage == 2:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_xsubsiminput.rotationAzimuthImage)
            rotationAzimuth = im.val
        else:
            rotationAzimuth = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationAzimuthValue, 0, rotationAzimuth)

        rotationDipLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_xsubsiminput.rotationDipImage)
            rotationDip = im.val
        else:
            rotationDip = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationDipValue, 0, rotationDip)

        rotationPlungeLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_xsubsiminput.rotationPlungeImage)
            rotationPlunge = im.val
        else:
            rotationPlunge = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationPlungeValue, 0, rotationPlunge)

    # searchNeighborhoodParameters
    searchNeighborhoodParameters = np.array(nv*[None])
    for i in range(nv):
        sn_c = deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_getitem(mpds_xsubsiminput.searchNeighborhoodParameters, i)
        sn = search_neighborhood_parameters_C2py(sn_c)
        if sn is None:
            print(f'ERROR ({fname}): can not convert search neighborhood parameters from C to python')
            return None
        searchNeighborhoodParameters[i] = sn

    # nneighboringNode
    nneighboringNode = np.zeros(nv, dtype='intc')
    deesse.mpds_get_array_from_int_vector(mpds_xsubsiminput.nneighboringNode, 0, nneighboringNode)
    nneighboringNode = nneighboringNode.astype('int')

    # maxPropInequalityNode
    maxPropInequalityNode = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.maxPropInequalityNode, 0, maxPropInequalityNode)
    maxPropInequalityNode = maxPropInequalityNode.astype('float')

    # neighboringNodeDensity
    neighboringNodeDensity = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.neighboringNodeDensity, 0, neighboringNodeDensity)
    neighboringNodeDensity = neighboringNodeDensity.astype('float')

    # simType
    simType_c = mpds_xsubsiminput.simAndPathParameters.simType
    if simType_c == deesse.SIM_ONE_BY_ONE:
        simType = 'sim_one_by_one'
    elif simType_c == deesse.SIM_VARIABLE_VECTOR:
        simType = 'sim_variable_vector'
    else:
        print(f'ERROR ({fname}): simulation type unknown')
        return None

    # simPathType
    simPathType = None
    simPathPower = None
    simPathStrength = None
    simPathUnilateralOrder = None

    simPathType_c = mpds_xsubsiminput.simAndPathParameters.pathType
    if simPathType_c == deesse.PATH_RANDOM:
        simPathType = 'random'
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_PDF:
        simPathType = 'random_hd_distance_pdf'
        simPathStrength = mpds_xsubsiminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SORT:
        simPathType = 'random_hd_distance_sort'
        simPathStrength = mpds_xsubsiminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SUM_PDF:
        simPathType = 'random_hd_distance_sum_pdf'
        simPathPower = mpds_xsubsiminput.simAndPathParameters.pow
        simPathStrength = mpds_xsubsiminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_RANDOM_HD_DISTANCE_SUM_SORT:
        simPathType = 'random_hd_distance_sum_sort'
        simPathPower = mpds_xsubsiminput.simAndPathParameters.pow
        simPathStrength = mpds_xsubsiminput.simAndPathParameters.strength
    elif simPathType_c == deesse.PATH_UNILATERAL:
        simPathType = 'unilateral'
        simPathUnilateralOrder = np.zeros(mpds_xsubsiminput.simAndPathParameters.unilateralOrderLength, dtype='intc')
        deesse.mpds_get_array_from_int_vector(mpds_xsubsiminput.simAndPathParameters.unilateralOrder, 0, simPathUnilateralOrder)
        simPathUnilateralOrder = simPathUnilateralOrder.astype('int')
    else:
        print(f'ERROR ({fname}): simulation path type unknown')
        return None

    # distanceThreshold
    distanceThreshold = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.distanceThreshold, 0, distanceThreshold)

    # softProbability
    softProbability = np.array(nv*[None])
    for i in range(nv):
        sp_c = deesse.MPDS_SOFTPROBABILITY_array_getitem(mpds_xsubsiminput.softProbability, i)
        sp = softProbability_C2py(sp_c)
        if sp is None:
            print(f'ERROR ({fname}): can not convert soft probability from C to python')
            return None
        softProbability[i] = sp

    # maxScanFraction
    maxScanFraction = np.zeros(nTI, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.maxScanFraction, 0, maxScanFraction)
    maxScanFraction = maxScanFraction.astype('float')

    # pyramidGeneralParameters
    pyramidGeneralParameters = pyramidGeneralParameters_C2py(mpds_xsubsiminput.pyramidGeneralParameters)
    if pyramidGeneralParameters is None:
        print(f'ERROR ({fname}): can not convert pyramid general parameters from C to python')
        return None

    # pyramidParameters
    pyramidParameters = np.array(nv*[None])
    for i in range(nv):
        pp_c = deesse.MPDS_PYRAMIDPARAMETERS_array_getitem(mpds_xsubsiminput.pyramidParameters, i)
        pp = pyramidParameters_C2py(pp_c)
        if pp is None:
            print(f'ERROR ({fname}): can not convert pyramid parameters from C to python')
            return None
        pyramidParameters[i] = pp

    # tolerance
    tolerance = mpds_xsubsiminput.tolerance

    # npostProcessingPathMax
    npostProcessingPathMax = mpds_xsubsiminput.npostProcessingPathMax

    # default parameters
    postProcessingNneighboringNode = None
    postProcessingNeighboringNodeDensity = None
    postProcessingDistanceThreshold = None
    postProcessingMaxScanFraction = None
    postProcessingTolerance = 0.0

    if npostProcessingPathMax > 0:
        # postProcessingNneighboringNode
        postProcessingNneighboringNode = np.zeros(nv, dtype='intc')
        deesse.mpds_get_array_from_int_vector(mpds_xsubsiminput.postProcessingNneighboringNode, 0, postProcessingNneighboringNode)
        postProcessingNneighboringNode = postProcessingNneighboringNode.astype('int')

        # postProcessingNeighboringNodeDensity
        postProcessingNeighboringNodeDensity = np.zeros(nv, dtype='double')
        deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.postProcessingNeighboringNodeDensity, 0, postProcessingNeighboringNodeDensity)
        postProcessingNeighboringNodeDensity = postProcessingNeighboringNodeDensity.astype('float')

        # postProcessingDistanceThreshold
        postProcessingDistanceThreshold = np.zeros(nv, dtype='float')
        deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.postProcessingDistanceThreshold, 0, postProcessingDistanceThreshold)

        # mpds_xsubsiminput.postProcessingMaxScanFraction
        postProcessingMaxScanFraction = np.zeros(nTI, dtype='double')
        deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.postProcessingMaxScanFraction, 0, postProcessingMaxScanFraction)
        postProcessingMaxScanFraction = postProcessingMaxScanFraction.astype('float')

        # mpds_xsubsiminput.postProcessingTolerance
        postProcessingTolerance = mpds_xsubsiminput.postProcessingTolerance

    section_parameters = DeesseXInputSection(
        nx=nx, ny=ny, nz=nz,
        nv=nv, distanceType=distanceType,
        sectionType=sectionType,
        nTI=nTI, TI=TI, simGridAsTiFlag=simGridAsTiFlag, pdfTI=pdfTI,
        homothetyUsage=homothetyUsage,
        homothetyXLocal=homothetyXLocal, homothetyXRatio=homothetyXRatio,
        homothetyYLocal=homothetyYLocal, homothetyYRatio=homothetyYRatio,
        homothetyZLocal=homothetyZLocal, homothetyZRatio=homothetyZRatio,
        rotationUsage=rotationUsage,
        rotationAzimuthLocal=rotationAzimuthLocal, rotationAzimuth=rotationAzimuth,
        rotationDipLocal=rotationDipLocal,         rotationDip=rotationDip,
        rotationPlungeLocal=rotationPlungeLocal,  rotationPlunge=rotationPlunge,
        searchNeighborhoodParameters=searchNeighborhoodParameters,
        nneighboringNode=nneighboringNode,
        maxPropInequalityNode=maxPropInequalityNode, neighboringNodeDensity=neighboringNodeDensity,
        simType=simType,
        simPathType=simPathType,
        simPathStrength=simPathStrength,
        simPathPower=simPathPower,
        simPathUnilateralOrder=simPathUnilateralOrder,
        distanceThreshold=distanceThreshold,
        softProbability=softProbability,
        maxScanFraction=maxScanFraction,
        pyramidGeneralParameters=pyramidGeneralParameters,
        pyramidParameters=pyramidParameters,
        tolerance=tolerance,
        npostProcessingPathMax=npostProcessingPathMax,
        postProcessingNneighboringNode=postProcessingNneighboringNode,
        postProcessingNeighboringNodeDensity=postProcessingNeighboringNodeDensity,
        postProcessingDistanceThreshold=postProcessingDistanceThreshold,
        postProcessingMaxScanFraction=postProcessingMaxScanFraction,
        postProcessingTolerance=postProcessingTolerance)

    return section_parameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_py2C(deesseX_input):
    """
    Converts deesseX input from python to C.

    :param deesseX_input: (DeesseXInput class) deesseX input - python
    :return:              (MPDS_XSIMINPUT *) deesseX input - C
    """

    fname = 'deesseX_input_py2C'

    nx = int(deesseX_input.nx)
    ny = int(deesseX_input.ny)
    nz = int(deesseX_input.nz)
    sx = float(deesseX_input.sx)
    sy = float(deesseX_input.sy)
    sz = float(deesseX_input.sz)
    ox = float(deesseX_input.ox)
    oy = float(deesseX_input.oy)
    oz = float(deesseX_input.oz)
    nv = int(deesseX_input.nv)

    # Allocate mpds_xsiminput
    mpds_xsiminput = deesse.malloc_MPDS_XSIMINPUT()

    # Init mpds_xsiminput
    deesse.MPDSInitXSimInput(mpds_xsiminput)

    # mpds_xsiminput.consoleAppFlag
    if deesseX_input.consoleAppFlag:
        mpds_xsiminput.consoleAppFlag = deesse.TRUE
    else:
        mpds_xsiminput.consoleAppFlag = deesse.FALSE

    # mpds_xsiminput.simName
    # (mpds_xsiminput.simName not used, but must be set (could be '')!
    if not isinstance(deesseX_input.simName, str):
        print(f'ERROR ({fname}): simName is not a string')
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        #deesse.MPDSFree(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        return None
    if len(deesseX_input.simName) >= deesse.MPDS_VARNAME_LENGTH:
        print(f'ERROR ({fname}): simName is too long')
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        #deesse.MPDSFree(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        return None

    deesse.mpds_x_allocate_and_set_simname(mpds_xsiminput, deesseX_input.simName)
    # mpds_xsiminput.simName = deesseX_input.simName #  works too

    # mpds_xsiminput.simImage ...
    # ... set initial image im (for simulation)
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=nv, val=deesse.MPDS_MISSING_VALUE,
             varname=deesseX_input.varname)

    # ... convert im from python to C
    mpds_xsiminput.simImage = img_py2C(im)

    # mpds_xsiminput.nvar
    mpds_xsiminput.nvar = nv

    # mpds_xsiminput.outputVarFlag
    deesse.mpds_x_allocate_and_set_outputVarFlag(mpds_xsiminput, np.array([int(i) for i in deesseX_input.outputVarFlag], dtype='bool'))

    # mpds_xsiminput.formatStringVar: not used

    # mpds_xsiminput.outputSimJob
    mpds_xsiminput.outputSimJob = deesse.OUTPUT_SIM_NO_FILE

    # mpds_xsiminput.outputSimImageFileName: not used (NULL: no output file!)

    # mpds_xsiminput.outputSectionTypeFlag
    if deesseX_input.outputSectionTypeFlag:
        mpds_xsiminput.outputSectionTypeFlag = deesse.TRUE
    else:
        mpds_xsiminput.outputSectionTypeFlag = deesse.FALSE

    # mpds_xsiminput.outputSectionTypeFileName: not used (NULL: no output file!)

    # mpds_xsiminput.outputSectionStepFlag
    if deesseX_input.outputSectionStepFlag:
        mpds_xsiminput.outputSectionStepFlag = deesse.TRUE
    else:
        mpds_xsiminput.outputSectionStepFlag = deesse.FALSE

    # mpds_xsiminput.outputSectionStepFileName: not used (NULL: no output file!)

    # mpds_xsiminput.outputReportFlag
    if deesseX_input.outputReportFlag:
        mpds_xsiminput.outputReportFlag = deesse.TRUE
        deesse.mpds_x_allocate_and_set_outputReportFileName(mpds_xsiminput, deesse_input.outputReportFileName)
    else:
        mpds_xsiminput.outputReportFlag = deesse.FALSE

    # mpds_xsiminput.ndataImage and mpds_xsiminput.dataImage
    if deesseX_input.dataImage is None:
        mpds_xsiminput.ndataImage = 0
    else:
        n = len(deesseX_input.dataImage)
        mpds_xsiminput.ndataImage = n
        mpds_xsiminput.dataImage = deesse.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(deesseX_input.dataImage):
            im_c = img_py2C(dataIm)
            deesse.MPDS_IMAGE_array_setitem(mpds_xsiminput.dataImage, i, im_c)
            # deesse.free_MPDS_IMAGE(im_c)
            #
            # deesse.MPDS_IMAGE_array_setitem(mpds_xsiminput.dataImage, i, img_py2C(dataIm))

    # mpds_xsiminput.ndataPointSet and mpds_xsiminput.dataPointSet
    if deesseX_input.dataPointSet is None:
        mpds_xsiminput.ndataPointSet = 0
    else:
        n = len(deesseX_input.dataPointSet)
        mpds_xsiminput.ndataPointSet = n
        mpds_xsiminput.dataPointSet = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesseX_input.dataPointSet):
            ps_c = ps_py2C(dataPS)
            deesse.MPDS_POINTSET_array_setitem(mpds_xsiminput.dataPointSet, i, ps_c)
            # deesse.free_MPDS_POINTSET(ps_c)
            #
            # deesse.MPDS_POINTSET_array_setitem(mpds_xsiminput.dataPointSet, i, ps_py2C(dataPS))

    # mpds_xsiminput.maskImageFlag and mpds_xsiminput.maskImage
    if deesseX_input.mask is None:
        mpds_xsiminput.maskImageFlag = deesse.FALSE
    else:
        mpds_xsiminput.maskImageFlag = deesse.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=deesseX_input.mask)
        mpds_xsiminput.maskImage = img_py2C(im)

    # mpds_xsiminput.trainValueRangeExtensionMax
    mpds_xsiminput.trainValueRangeExtensionMax = deesseX_input.expMax

    # mpds_xsiminput.normalizingType
    normalizingType_dict = {
        'linear'  : deesse.NORMALIZING_LINEAR,
        'uniform' : deesse.NORMALIZING_UNIFORM,
        'normal'  : deesse.NORMALIZING_NORMAL
    }
    try:
        mpds_xsiminput.normalizingType = normalizingType_dict[deesseX_input.normalizingType]
    except:
        print(f'ERROR ({fname}): normalizing type unknown')
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        #deesse.MPDSFree(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        return None

    # mpds_xsimInput.rescalingMode
    rescalingMode_dict = {
        'none'        : deesse.RESCALING_NONE,
        'min_max'     : deesse.RESCALING_MIN_MAX,
        'mean_length' : deesse.RESCALING_MEAN_LENGTH
    }
    mpds_xsiminput.rescalingMode = deesse.new_MPDS_RESCALINGMODE_array(nv)
    for i, m in enumerate(deesseX_input.rescalingMode):
        if m in rescalingMode_dict.keys():
            deesse.MPDS_RESCALINGMODE_array_setitem(mpds_xsiminput.rescalingMode, i, rescalingMode_dict[m])
        else:
            print(f'ERROR ({fname}): rescaling mode unknown')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None

    # mpds_xsimInput.rescalingTargetMin
    mpds_xsiminput.rescalingTargetMin = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsiminput.rescalingTargetMin, 0,
        np.asarray(deesseX_input.rescalingTargetMin).reshape(nv))

    # mpds_xsimInput.rescalingTargetMax
    mpds_xsiminput.rescalingTargetMax = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsiminput.rescalingTargetMax, 0,
        np.asarray(deesseX_input.rescalingTargetMax).reshape(nv))

    # mpds_xsimInput.rescalingTargetMean
    mpds_xsiminput.rescalingTargetMean = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsiminput.rescalingTargetMean, 0,
        np.asarray(deesseX_input.rescalingTargetMean).reshape(nv))

    # mpds_xsiminput.rescalingTargetLength
    mpds_xsiminput.rescalingTargetLength = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsiminput.rescalingTargetLength, 0,
        np.asarray(deesseX_input.rescalingTargetLength).reshape(nv))

    # mpds_xsiminput.relativeDistanceFlag
    deesse.mpds_x_allocate_and_set_relativeDistanceFlag(mpds_xsiminput, np.array([int(i) for i in deesseX_input.relativeDistanceFlag], dtype='bool')) # , dtype='intc'))

    # mpds_xsiminput.distanceType
    mpds_xsiminput.distanceType = deesse.new_int_array(nv)
    deesse.mpds_set_int_vector_from_array(
        mpds_xsiminput.distanceType, 0,
        np.asarray(deesseX_input.distanceType, dtype='intc').reshape(nv))

    # mpds_xsiminput.powerLpDistance
    mpds_xsiminput.powerLpDistance = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsiminput.powerLpDistance, 0,
        np.asarray(deesseX_input.powerLpDistance).reshape(nv))

    # mpds_xsiminput.powerLpDistanceInv
    mpds_xsiminput.powerLpDistanceInv = deesse.new_double_array(nv)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsiminput.powerLpDistanceInv, 0,
        np.asarray(deesseX_input.powerLpDistanceInv).reshape(nv))

    # mpds_xsiminput.conditioningWeightFactor
    mpds_xsiminput.conditioningWeightFactor = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsiminput.conditioningWeightFactor, 0,
        np.asarray(deesseX_input.conditioningWeightFactor).reshape(nv))

    # mpds_xsiminput.XSectionParameters
    mpds_xsiminput.XSectionParameters = deesseX_input_sectionPath_py2C(deesseX_input.sectionPath_parameters)

    # mpds_xsiminput.XSectionParameters = deesse.malloc_MPDS_XSECTIONPARAMETERS()
    # deesse.MPDSInitXSectionParameters(mpds_xsiminput.XSectionParameters)
    #
    # # ... XSectionMode
    # sectionMode_dict = {
    #     'section_xy_xz_yz' : deesse.SECTION_XY_XZ_YZ,
    #     'section_xy_yz_xz' : deesse.SECTION_XY_YZ_XZ,
    #     'section_xz_xy_yz' : deesse.SECTION_XZ_XY_YZ,
    #     'section_xz_yz_xy' : deesse.SECTION_XZ_YZ_XY,
    #     'section_yz_xy_xz' : deesse.SECTION_YZ_XY_XZ,
    #     'section_yz_xz_xy' : deesse.SECTION_YZ_XZ_XY,
    #     'section_xy_xz'    : deesse.SECTION_XY_XZ,
    #     'section_xz_xy'    : deesse.SECTION_XZ_XY,
    #     'section_xy_yz'    : deesse.SECTION_XY_YZ,
    #     'section_yz_xy'    : deesse.SECTION_YZ_XY,
    #     'section_xz_yz'    : deesse.SECTION_XZ_YZ,
    #     'section_yz_xz'    : deesse.SECTION_YZ_XZ,
    #     'section_xy_z'     : deesse.SECTION_XY_Z,
    #     'section_z_xy'     : deesse.SECTION_Z_XY,
    #     'section_xz_y'     : deesse.SECTION_XZ_Y,
    #     'section_y_xz'     : deesse.SECTION_Y_XZ,
    #     'section_yz_x'     : deesse.SECTION_YZ_X,
    #     'section_x_yz'     : deesse.SECTION_X_YZ,
    #     'section_x_y_z'    : deesse.SECTION_X_Y_Z,
    #     'section_x_z_y'    : deesse.SECTION_X_Z_Y,
    #     'section_y_x_z'    : deesse.SECTION_Y_X_Z,
    #     'section_y_z_x'    : deesse.SECTION_Y_Z_X,
    #     'section_z_x_y'    : deesse.SECTION_Z_X_Y,
    #     'section_z_y_x'    : deesse.SECTION_Z_Y_X,
    #     'section_x_y'      : deesse.SECTION_X_Y,
    #     'section_y_x'      : deesse.SECTION_Y_X,
    #     'section_x_z'      : deesse.SECTION_X_Z,
    #     'section_z_x'      : deesse.SECTION_Z_X,
    #     'section_y_z'      : deesse.SECTION_Y_Z,
    #     'section_z_y'      : deesse.SECTION_Z_Y
    # }
    # try:
    #     mpds_xsiminput.XSectionParameters.XSectionMode = sectionMode_dict[deesseX_input.sectionPath_parameters.sectionMode]
    # except:
    #     print(f'ERROR ({fname}): section mode unknown')
    #     # Free memory on C side
    #     deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #     #deesse.MPDSFree(mpds_xsiminput)
    #     deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
    #     return None
    #
    # # ... XSectionPathMode and other relevant fields
    # if deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_random':
    #     mpds_xsiminput.XSectionParameters.XSectionPathMode = deesse.SECTION_PATH_RANDOM
    #
    # elif deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_pow_2':
    #     mpds_xsiminput.XSectionParameters.XSectionPathMode = deesse.SECTION_PATH_POW_2
    #
    # elif deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_subdiv':
    #     mpds_xsiminput.XSectionParameters.XSectionPathMode = deesse.SECTION_PATH_SUBDIV
    #     mpds_xsiminput.XSectionParameters.minSpaceX = deesseX_input.sectionPath_parameters.minSpaceX
    #     mpds_xsiminput.XSectionParameters.minSpaceY = deesseX_input.sectionPath_parameters.minSpaceY
    #     mpds_xsiminput.XSectionParameters.minSpaceZ = deesseX_input.sectionPath_parameters.minSpaceZ
    #     if deesseX_input.sectionPath_parameters.balancedFillingFlag:
    #         mpds_xsiminput.XSectionParameters.balancedFillingFlag = deesse.TRUE
    #     else:
    #         mpds_xsiminput.XSectionParameters.balancedFillingFlag = deesse.FALSE
    #
    # elif deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_manual':
    #     mpds_xsiminput.XSectionParameters.XSectionPathMode = deesse.SECTION_PATH_MANUAL
    #     ns = int(deesseX_input.sectionPath_parameters.nsection)
    #     mpds_xsiminput.XSectionParameters.nsection = ns
    #     if ns > 0:
    #         mpds_xsiminput.XSectionParameters.sectionType = deesse.new_int_array(ns)
    #         deesse.mpds_set_int_vector_from_array(
    #             mpds_siminput.XSectionParameters.sectionType, 0,
    #             np.asarray(deesseX_input.sectionPath_parameters.sectionType, dtype='intc').reshape(ns))
    #         mpds_xsiminput.XSectionParameters.sectionLoc = deesse.new_int_array(ns)
    #         deesse.mpds_set_int_vector_from_array(
    #             mpds_siminput.XSectionParameters.sectionLoc, 0,
    #             np.asarray(deesseX_input.sectionPath_parameters.sectionLoc, dtype='intc').reshape(ns))
    # else:
    #     print(f'ERROR ({fname}): section path type unknown')
    #     # Free memory on C side
    #     deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #     #deesse.MPDSFree(mpds_xsiminput)
    #     deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
    #     return None

    # mpds_xsiminput.XSubSimInput_<*> ...
    for sect_param in deesseX_input.section_parameters:
        if sect_param.nx != nx:
            print(f'ERROR ({fname}): nx in (one) section parameters invalid')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None
        if sect_param.ny != ny:
            print(f'ERROR ({fname}): ny in (one) section parameters invalid')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None
        if sect_param.nz != nz:
            print(f'ERROR ({fname}): nz in (one) section parameters invalid')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None
        if sect_param.nv != nv:
            print(f'ERROR ({fname}): nv in (one) section parameters invalid')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None
        if not np.all(deesseX_input.distanceType == sect_param.distanceType):
            print(f"ERROR ({fname}): 'distanceType' (one) section parameters invalid")
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None
        # for d1, d2 in zip(deesseX_input.distanceType, sect_param.distanceType):
        #     if d1 != d2:
        #         print("ERROR: 'distanceType' (one) section parameters invalid")
        #         # Free memory on C side
        #         deesse.MPDSFreeXSimInput(mpds_xsiminput)
        #         #deesse.MPDSFree(mpds_xsiminput)
        #         deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        #         return None
        if sect_param.sectionType == 0:
            mpds_xsiminput.XSubSimInput_xy = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                        nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        elif sect_param.sectionType == 1:
            mpds_xsiminput.XSubSimInput_xz = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                        nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        elif sect_param.sectionType == 2:
            mpds_xsiminput.XSubSimInput_yz = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                        nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        elif sect_param.sectionType == 3:
            mpds_xsiminput.XSubSimInput_z = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                       nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        elif sect_param.sectionType == 4:
            mpds_xsiminput.XSubSimInput_y = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                       nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        elif sect_param.sectionType == 5:
            mpds_xsiminput.XSubSimInput_x = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                       nx, ny, nz, sx, sy, sz, ox, oy, oz, nv)
        else:
            print(f'ERROR ({fname}): section type in section parameters unknown')
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            #deesse.MPDSFree(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            return None

    # mpds_xsiminput.seed
    mpds_xsiminput.seed = int(deesseX_input.seed)

    # mpds_xsiminput.seedIncrement
    mpds_xsiminput.seedIncrement = int(deesseX_input.seedIncrement)

    # mpds_xsiminput.nrealization
    mpds_xsiminput.nrealization = int(deesseX_input.nrealization)

    return mpds_xsiminput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_C2py(mpds_xsiminput):
    """
    Converts deesseX input from C to python.

    :param mpds_xsiminput:  (MPDS_XSIMINPUT *) deesseX input - C
    :return:                (DeesseInput class) deesseX input - python
    """

    fname = 'deesseX_input_C2py'

    # simName
    simName = mpds_xsiminput.simName

    im = img_C2py(mpds_xsiminput.simImage)

    # grid geometry
    nx = im.nx
    ny = im.ny
    nz = im.nz
    sx = im.sx
    sy = im.sy
    sz = im.sz
    ox = im.ox
    oy = im.oy
    oz = im.oz

    # nv (number of variable(s))
    nv = im.nv # or: nv = int(mpds_xsiminput.nvar)

    # varname
    varname = im.varname

    # outputVarFlag
    outputVarFlag = np.zeros(nv, dtype='intc')
    deesse.mpds_x_get_outputVarFlag(mpds_xsiminput, outputVarFlag)
    outputVarFlag = outputVarFlag.astype('bool')

    # output maps
    outputSectionTypeFlag  = bool(int.from_bytes(mpds_xsiminput.outputSectionTypeFlag.encode('utf-8'), byteorder='big'))
    outputSectionStepFlag  = bool(int.from_bytes(mpds_xsiminput.outputSectionStepFlag.encode('utf-8'), byteorder='big'))
    outputReportFlag       = bool(int.from_bytes(mpds_xsiminput.outputReportFlag.encode('utf-8'), byteorder='big'))

    # report
    if outputReportFlag:
        outputReportFileName = mpds_xsiminput.outputReportFileName
    else:
        outputReportFileName = None

    # dataImage
    dataImage = None
    ndataImage = mpds_xsiminput.ndataImage
    if ndataImage > 0:
        dataImage = np.array(ndataImage*[None])
        for i in range(ndataImage):
            im = deesse.MPDS_IMAGE_array_getitem(mpds_xsiminput.dataImage, i)
            dataImage[i] = img_C2py(im)

    # dataPointSet
    dataPointSet = None
    ndataPointSet = mpds_xsiminput.ndataPointSet
    if ndataPointSet > 0:
        dataPointSet = np.array(ndataPointSet*[None])
        for i in range(ndataPointSet):
            ps = deesse.MPDS_POINTSET_array_getitem(mpds_xsiminput.dataPointSet, i)
            dataPointSet[i] = ps_C2py(ps)

    # mask
    mask = None
    flag = bool(int.from_bytes(mpds_xsiminput.maskImageFlag.encode('utf-8'), byteorder='big'))
    if flag:
        im = img_C2py(mpds_xsiminput.maskImage)
        mask = im.val

    # expMax
    expMax = mpds_xsiminput.trainValueRangeExtensionMax

    # normalizingType
    normalizingType_dict = {
        deesse.NORMALIZING_LINEAR  : 'linear',
        deesse.NORMALIZING_UNIFORM : 'uniform',
        deesse.NORMALIZING_NORMAL  : 'normal'
    }
    try:
        normalizingType = normalizingType_dict[mpds_xsiminput.normalizingType]
    except:
        print(f'ERROR ({fname}): normalizing type unknown')
        return None

    # rescalingMode
    rescalingMode_dict = {
        deesse.RESCALING_NONE        : 'none',
        deesse.RESCALING_MIN_MAX     : 'min_max',
        deesse.RESCALING_MEAN_LENGTH : 'mean_length'
    }
    rescalingMode = np.array(nv*[None])
    for i in range(nv):
        rs_c = deesse.MPDS_RESCALINGMODE_array_getitem(mpds_xsiminput.rescalingMode, i)
        try:
            rs = rescalingMode_dict[rs_c]
            rescalingMode[i] = rs
        except:
            print(f'ERROR ({fname}): rescaling mode unknown')
            return None

    # rescalingTargetMin
    rescalingTargetMin = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsiminput.rescalingTargetMin, 0, rescalingTargetMin)

    # rescalingTargetMax
    rescalingTargetMax = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsiminput.rescalingTargetMax, 0, rescalingTargetMax)

    # rescalingTargetMean
    rescalingTargetMean = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsiminput.rescalingTargetMean, 0, rescalingTargetMean)

    # rescalingTargetLength
    rescalingTargetLength = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsiminput.rescalingTargetLength, 0, rescalingTargetLength)

    # relativeDistanceFlag
    relativeDistanceFlag = np.zeros(nv, dtype='intc')
    deesse.mpds_x_get_relativeDistanceFlag(mpds_xsiminput, relativeDistanceFlag)
    relativeDistanceFlag = relativeDistanceFlag.astype('bool')

    # distanceType
    distanceType = np.zeros(nv, dtype='intc')
    deesse.mpds_get_array_from_int_vector(mpds_xsiminput.distanceType, 0, distanceType)
    distanceType = distanceType.astype('int')
    distanceType = list(distanceType)
    for i in range(nv):
        if distanceType[i] == 0:
            distanceType[i] = 'categorical'
        elif distanceType[i] == 1:
            distanceType[i] = 'continuous'

    # powerLpDistance
    powerLpDistance = np.zeros(nv, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_xsiminput.powerLpDistance, 0, powerLpDistance)
    powerLpDistance = powerLpDistance.astype('float')
    for i in range(nv):
        if distanceType[i] != 3:
            powerLpDistance[i] = 1.0

    # conditioningWeightFactor
    conditioningWeightFactor = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsiminput.conditioningWeightFactor, 0, conditioningWeightFactor)

    # sectionPath_parameters
    sectionPath_parameters = deesseX_input_sectionPath_C2py(mpds_xsiminput.XSectionParameters)

    # section_parameters
    section_parameters = []
    if sectionPath_parameters.sectionMode in (
            'section_xy_xz_yz',
            'section_xy_yz_xz',
            'section_xz_xy_yz',
            'section_xz_yz_xy',
            'section_yz_xy_xz',
            'section_yz_xz_xy',
            'section_xy_xz',
            'section_xz_xy',
            'section_xy_yz',
            'section_yz_xy',
            'section_xy_z',
            'section_z_xy'):
        sectionType = 0
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_xy, sectionType, nx, ny, nz, nv, distanceType))

    if sectionPath_parameters.sectionMode in (
            'section_xy_xz_yz',
            'section_xy_yz_xz',
            'section_xz_xy_yz',
            'section_xz_yz_xy',
            'section_yz_xy_xz',
            'section_yz_xz_xy',
            'section_xy_xz',
            'section_xz_xy',
            'section_xz_yz',
            'section_yz_xz',
            'section_xz_y',
            'section_y_xz'):
        sectionType = 1
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_xz, sectionType, nx, ny, nz, nv, distanceType))

    if sectionPath_parameters.sectionMode in (
            'section_xy_xz_yz',
            'section_xy_yz_xz',
            'section_xz_xy_yz',
            'section_xz_yz_xy',
            'section_yz_xy_xz',
            'section_yz_xz_xy',
            'section_xy_yz',
            'section_yz_xy',
            'section_xz_yz',
            'section_yz_xz',
            'section_yz_x',
            'section_x_yz'):
        sectionType = 2
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_yz, sectionType, nx, ny, nz, nv, distanceType))

    if sectionPath_parameters.sectionMode in (
            'section_xy_z',
            'section_z_xy',
            'section_x_y_z',
            'section_x_z_y',
            'section_y_x_z',
            'section_y_z_x',
            'section_z_x_y',
            'section_z_y_x',
            'section_x_z',
            'section_z_x',
            'section_y_z',
            'section_z_y'):
        sectionType = 3
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_z, sectionType, nx, ny, nz, nv, distanceType))

    if sectionPath_parameters.sectionMode in (
            'section_xz_y',
            'section_y_xz',
            'section_x_y_z',
            'section_x_z_y',
            'section_y_x_z',
            'section_y_z_x',
            'section_z_x_y',
            'section_z_y_x',
            'section_x_y',
            'section_y_x',
            'section_y_z',
            'section_z_y'):
        sectionType = 4
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_y, sectionType, nx, ny, nz, nv, distanceType))

    if sectionPath_parameters.sectionMode in (
            'section_yz_x',
            'section_x_yz',
            'section_x_y_z',
            'section_x_z_y',
            'section_y_x_z',
            'section_y_z_x',
            'section_z_x_y',
            'section_z_y_x',
            'section_x_y',
            'section_y_x',
            'section_x_z',
            'section_z_x'):
        sectionType = 5
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_x, sectionType, nx, ny, nz, nv, distanceType))

    section_parameters = np.asarray(section_parameters)

    # seed
    seed = mpds_xsiminput.seed

    # seedIncrement
    seedIncrement = mpds_xsiminput.seedIncrement

    # nrealization
    nrealization = mpds_xsiminput.nrealization

    # deesseX input
    deesseX_input = DeesseXInput(
        simName=simName,
        nx=nx, ny=ny, nz=nz,
        sx=sx, sy=sy, sz=sz,
        ox=ox, oy=oy, oz=oz,
        nv=nv, varname=varname, outputVarFlag=outputVarFlag,
        outputSectionTypeFlag=outputSectionTypeFlag,
        outputSectionStepFlag=outputSectionStepFlag,
        outputReportFlag=outputReportFlag, outputReportFileName=outputReportFileName,
        dataImage=dataImage, dataPointSet=dataPointSet,
        mask=mask,
        expMax=expMax,
        normalizingType=normalizingType,
        rescalingMode=rescalingMode,
        rescalingTargetMin=rescalingTargetMin, rescalingTargetMax=rescalingTargetMax,
        rescalingTargetMean=rescalingTargetMean, rescalingTargetLength=rescalingTargetLength,
        relativeDistanceFlag=relativeDistanceFlag,
        distanceType=distanceType,
        powerLpDistance=powerLpDistance,
        conditioningWeightFactor=conditioningWeightFactor,
        sectionPath_parameters=sectionPath_parameters,
        section_parameters=section_parameters,
        seed=seed,
        seedIncrement=seedIncrement,
        nrealization=nrealization)

    return deesseX_input
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_output_C2py(mpds_xsimoutput, mpds_progressMonitor):
    """
    Get deesseX output for python from C.

    :param mpds_xsimoutput: (MPDS_XSIMOUTPUT *) simulation output - (C struct)
                                contains output of deesseX simulation
    :param mpds_progressMonitor:
                            (MPDS_PROGRESSMONITOR *) progress monitor - (C struct)
                                contains output messages (warnings) of deesseX
                                simulation

    :return deesseX_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'simSectionType':simSectionType,
             'simSectionStep':simSectionStep,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = mpds_xsimoutput->nreal (number of realizations):

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_xsimoutput->outputSimImage[0])
                    (sim is None if mpds_xsimoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_xsimoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_xsimoutput->originalVarIndex is NULL)

        simSectionType:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionType[i]: section type (id identifying which type of
                        section is used) map for the i-th realisation
                        (mpds_xsimoutput->outputSectionTypeImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionType may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionType is None if
                    mpds_xsimoutput->outputSectionTypeImage is NULL)

        simSectionStep:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionStep[i]: section step (index of simulation by
                        direct sampling of (a bunch of) sections of same type)
                        map for the i-th realisation,
                        (mpds_xsimoutput->outputSectionStepImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionStep may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionStep is None if
                    mpds_xsimoutput->outputSectionStepImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # fname = 'deesseX_output_C2py'

    # Initialization
    sim, sim_var_original_index = None, None
    simSectionType, simSectionStep = None, None
    nwarning, warnings = None, None

    if mpds_xsimoutput.nreal:
        nreal = mpds_xsimoutput.nreal

        if mpds_xsimoutput.nvarSimPerReal:
            # --- sim_var_original_index ---
            sim_var_original_index = np.zeros(mpds_xsimoutput.nvarSimPerReal, dtype='intc') # 'intc' for C-compatibility
            deesse.mpds_get_array_from_int_vector(mpds_xsimoutput.originalVarIndex, 0, sim_var_original_index)

            # ... also works ...
            # sim_var_original_index = np.asarray([deesse.int_array_getitem(mpds_xsimoutput.originalVarIndex, i) for i in range(mpds_xsimoutput.nvarSimPerReal)])
            # ...
            # ---

            # --- sim ---
            im = img_C2py(mpds_xsimoutput.outputSimImage)

            nv = mpds_xsimoutput.nvarSimPerReal
            k = 0
            sim = []
            for i in range(nreal):
                sim.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                               sx=im.sx, sy=im.sy, sz=im.sz,
                               ox=im.ox, oy=im.oy, oz=im.oz,
                               nv=nv, val=im.val[k:(k+nv),...],
                               varname=im.varname[k:(k+nv)]))
                k = k + nv

            del(im)
            sim = np.asarray(sim).reshape(nreal)
            # ---

        if mpds_xsimoutput.nvarSectionType:
            # --- simSectionType ---
            im = img_C2py(mpds_xsimoutput.outputSectionTypeImage)

            nv = mpds_xsimoutput.nvarSectionType
            simSectionType = []
            for i in range(nv):
                simSectionType.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                          sx=im.sx, sy=im.sy, sz=im.sz,
                                          ox=im.ox, oy=im.oy, oz=im.oz,
                                          nv=1, val=im.val[i,...],
                                          varname=im.varname[i]))

            del(im)
            simSectionType = np.asarray(simSectionType).reshape(nv)
            # ---

        if mpds_xsimoutput.nvarSectionStep:
            # --- simSectionStep ---
            im = img_C2py(mpds_xsimoutput.outputSectionStepImage)

            nv = mpds_xsimoutput.nvarSectionStep
            simSectionStep = []
            for i in range(nv):
                simSectionStep.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                          sx=im.sx, sy=im.sy, sz=im.sz,
                                          ox=im.ox, oy=im.oy, oz=im.oz,
                                          nv=1, val=im.val[i,...],
                                          varname=im.varname[i]))

            del(im)
            simSectionStep = np.asarray(simSectionStep).reshape(nv)
            # ---

    # --- nwarning, warnings ---
    nwarning = mpds_progressMonitor.nwarning
    warnings = []
    if mpds_progressMonitor.nwarningNumber:
        tmp = np.zeros(mpds_progressMonitor.nwarningNumber, dtype='intc') # 'intc' for C-compatibility
        deesse.mpds_get_array_from_int_vector(mpds_progressMonitor.warningNumberList, 0, tmp)
        warningNumberList = np.asarray(tmp, dtype='int') # 'int' or equivalently 'int64'
        for iwarn in warningNumberList:
            warning_message = deesse.mpds_get_warning_message(int(iwarn)) # int() required!
            warning_message = warning_message.replace('\n', '')
            warnings.append(warning_message)
    # ---

    return {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'simSectionType':simSectionType, 'simSectionStep':simSectionStep,
        'nwarning':nwarning, 'warnings':warnings
        }
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseXRun(deesseX_input, nthreads=-1, verbose=2):
    """
    Launches deesseX.

    :param deesseX_input:
                (DeesseXInput (class)): deesseX input parameter (python)
    :param nthreads:
                (int) number of thread(s) to use for deesseX (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the deesseX run:
                    - 0: mininal display
                    - 1: only errors
                    - 2: version and warning(s) encountered
                    - 3 (or >2): version, progress, and warning(s) encountered

    :return deesseX_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'simSectionType':simSectionType,
             'simSectionStep':simSectionStep,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = deesseX_input.nrealization:

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_xsimoutput->outputSimImage[0])
                    (sim is None if mpds_xsimoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_xsimoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_xsimoutput->originalVarIndex is NULL)

        simSectionType:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionType[i]: section type (id identifying which type of
                        section is used) map for the i-th realisation
                        (mpds_xsimoutput->outputSectionTypeImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionType may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionType is None if
                    mpds_xsimoutput->outputSectionTypeImage is NULL)

        simSectionStep:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionStep[i]: section step (index of simulation by
                        direct sampling of (a bunch of) sections of same type)
                        map for the i-th realisation,
                        (mpds_xsimoutput->outputSectionStepImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionStep may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionStep is None if
                    mpds_xsimoutput->outputSectionStepImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    fname = 'deesseXRun'

    if not deesseX_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesseX input')
        return None

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('DeeSseX running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(deesse.MPDS_X_VERSION_NUMBER, deesse.MPDS_X_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesseX...

    # Convert deesseX input from python to C
    try:
        mpds_xsiminput = deesseX_input_py2C(deesseX_input)
    except:
        print(f'ERROR ({fname}): unable to convert deesseX input from python to C...')
        return None

    if mpds_xsiminput is None:
        print(f'ERROR ({fname}): unable to convert deesseX input from python to C...')
        return None

    # Allocate mpds_xsimoutput
    mpds_xsimoutput = deesse.malloc_MPDS_XSIMOUTPUT()

    # Init mpds_simoutput
    deesse.MPDSInitXSimOutput(mpds_xsimoutput)

    # Set progress monitor
    mpds_progressMonitor = deesse.malloc_MPDS_PROGRESSMONITOR()
    deesse.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to deesse.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = deesse.MPDSUpdateProgressMonitor4_ptr

    # Launch deesseX
    # err = deesse.MPDSXSim(mpds_xsiminput, mpds_xsimoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = deesse.MPDSOMPXSim(mpds_xsiminput, mpds_xsimoutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: deesseX input
    deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #deesse.MPDSFree(mpds_xsiminput)
    deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)

    if err:
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        deesseX_output = None
    else:
        deesseX_output = deesseX_output_C2py(mpds_xsimoutput, mpds_progressMonitor)

    # Free memory on C side: simulation output
    deesse.MPDSFreeXSimOutput(mpds_xsimoutput)
    #deesse.MPDSFree (mpds_xsimoutput)
    deesse.free_MPDS_XSIMOUTPUT(mpds_xsimoutput)

    # Free memory on C side: progress monitor
    #deesse.MPDSFree(mpds_progressMonitor)
    deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and deesseX_output:
        print('DeeSseX run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and deesseX_output and deesseX_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(deesseX_output['nwarning']))
        for i, warning_message in enumerate(deesseX_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return deesseX_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseXRun_mp(deesseX_input, nproc=None, nthreads_per_proc=None, verbose=2):
    """
    Launches deesseX through multiple processes.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function deesseXRun],
        - the set of realizations (specified by deesseX_input.nrealization) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param deesseX_input:
                (DeesseXInput (class)): deesseX input parameter (python)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count()), and
                    nreal is the number of realization
                    (deesseX_input.nrealization)

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: mininal display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return deesseX_output:
        (dict)
            {'sim':sim,
             'sim_var_original_index':sim_var_original_index,
             'simSectionType':simSectionType,
             'simSectionStep':simSectionStep,
             'nwarning':nwarning,
             'warnings':warnings}

        With nreal = deesseX_input.nrealization:

        sim:    (1-dimensional array of Img (class) of size nreal or None)
                    sim[i]: i-th realisation,
                        k-th variable stored refers to
                            - the original variable sim_var_original_index[k]
                        (get from mpds_xsimoutput->outputSimImage[0])
                    (sim is None if mpds_xsimoutput->outputSimImage is NULL)

        sim_var_original_index:
                (1-dimensional array of ints or None)
                    sim_var_original_index[k]: index of the original variable
                        (given in deesse_input) of the k-th variable stored in
                        in sim[i] for any i
                        (array of length array of length sim[*].nv,
                        get from mpds_xsimoutput->originalVarIndex)
                    (sim_var_original_index is None if
                    mpds_xsimoutput->originalVarIndex is NULL)

        simSectionType:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionType[i]: section type (id identifying which type of
                        section is used) map for the i-th realisation
                        (mpds_xsimoutput->outputSectionTypeImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionType may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionType is None if
                    mpds_xsimoutput->outputSectionTypeImage is NULL)

        simSectionStep:
                (1-dimensional array of Img (class) of size nreal, or 1 or None)
                    simSectionStep[i]: section step (index of simulation by
                        direct sampling of (a bunch of) sections of same type)
                        map for the i-th realisation,
                        (mpds_xsimoutput->outputSectionStepImage[0]); note:
                        depending on section path mode (see class DeesseXInput:
                        deesseX_input.sectionPath_parameters.sectionPathMode)
                        that was used, simSectionStep may be of size 1 even if
                        nreal is greater than 1, in such a case the same map is
                        valid for all realisations
                    (simSectionStep is None if
                    mpds_xsimoutput->outputSectionStepImage is NULL)

        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)

        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    fname = 'deesseXRun_mp'

    if not deesseX_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesseX input')
        return None

    if deesseX_input.nrealization <= 1:
        if verbose > 0:
            print('NOTE: number of realization does not exceed 1: launching deesseXRun...')
        nthreads = nthreads_per_proc
        if nthreads is None:
            nthreads = -1
        deesseX_output = deesseXRun(deesseX_input, nthreads=nthreads, verbose=verbose)
        return deesseX_output

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, deesseX_input.nrealization), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), deesseX_input.nrealization), 1)
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
    q, r = np.divmod(deesseX_input.nrealization, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('DeeSseX running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, deesse.MPDS_X_VERSION_NUMBER, deesse.MPDS_X_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesseX...

    # Initialize deesseX input for each process
    deesseX_input_proc = [copy.copy(deesseX_input) for i in range(nproc)]
    init_seed = deesseX_input.seed

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i, input in enumerate(deesseX_input_proc):
        # Adapt deesseX input for i-th process
        input.nrealization = real_index_proc[i+1] - real_index_proc[i]
        input.seed = init_seed + int(real_index_proc[i]) * input.seedIncrement
        input.outputReportFileName = input.outputReportFileName + f'.{i}'
        if i==0:
            verb = min(verbose, 1) # allow to print error for process i
        else:
            verb = 0
        # Launch deesseX (i-th process)
        out_pool.append(pool.apply_async(deesseXRun, args=(input, nth, verb)))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    deesseX_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in deesseX_output_proc]):
        return None

    sim, sim_var_original_index = None, None
    simSectionType, simSectionStep = None, None
    nwarning, warnings = None, None

    # Gather results from every process
    # sim
    sim = np.hstack([out['sim'] for out in deesseX_output_proc])
    # ... remove None entries
    sim = sim[[x is not None for x in sim]]
    # ... set to None if every entry is None
    if np.all([x is None for x in sim]):
        sim = None

    # sim_var_original_index
    sim_var_original_index = deesseX_output_proc[0]['sim_var_original_index']

    # simSectionType
    simSectionType = np.hstack([out['simSectionType'] for out in deesseX_output_proc])
    # ... remove None entries
    simSectionType = simSectionType[[x is not None for x in simSectionType]]
    # ... set to None if every entry is None
    if np.all([x is None for x in simSectionType]):
        simSectionType = None

    # simSectionStep
    simSectionStep = np.hstack([out['simSectionStep'] for out in deesseX_output_proc])
    # ... remove None entries
    simSectionStep = simSectionStep[[x is not None for x in simSectionStep]]
    # ... set to None if every entry is None
    if np.all([x is None for x in simSectionStep]):
        simSectionStep = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in deesseX_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in deesseX_output_proc])))

    # Adjust variable names
    ndigit = deesse.MPDS_X_NB_DIGIT_FOR_REALIZATION_NUMBER
    if sim is not None:
        for i in range(deesseX_input.nrealization):
            for k in range(sim[i].nv):
                sim[i].varname[k] = sim[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
    if simSectionType is not None:
        if deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_random':
            for i in range(deesse_input.nrealization):
                for k in range(simSectionType[i].nv):
                    simSectionType[i].varname[k] = simSectionType[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
        else: # keep only first map (all are the same)
            simSectionType = np.array([simSectionType[0]])
    if simSectionStep is not None:
        if deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_random':
            for i in range(deesse_input.nrealization):
                for k in range(simSectionStep[i].nv):
                    simSectionStep[i].varname[k] = simSectionStep[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
        else: # keep only first map (all are the same)
            simSectionStep = np.array([simSectionStep[0]])

    deesseX_output = {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'simSectionType':simSectionType, 'simSectionStep':simSectionStep,
        'nwarning':nwarning, 'warnings':warnings
        }

    if verbose >= 2 and deesseX_output:
        print('DeeSseX run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and deesseX_output and deesseX_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(deesseX_output['nwarning']))
        for i, warning_message in enumerate(deesseX_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return deesseX_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def exportDeesseXInput(
        deesseX_input,
        dirname='input_ascii',
        fileprefix='dsX',
        endofline='\n',
        verbose=2):
    """
    Exports deesseX input as ASCII files (in the directory named <dirname>).
    The command line version of deesseX can then be launched from the directory
    <dirname> by using the generated ASCII files.

    :param deesseX_input:   (DeesseXInput class) deesseX input - python
    :param dirname:         (string) name of the directory in which the files
                                will be written; if not existing, it will be
                                created;
                                WARNING: the generated files might erase already
                                existing ones!
    :param fileprefix:      (string) prefix for generated files, the main input
                                file will be <dirname>/<fileprefix>.in
    :param endofline:       (string) end of line string to be used for the
                                deesseX input file
    :param verbose:         (int) indicates which degree of detail is used when
                                writing comments in the deesseX input file
                                - 0: no comment
                                - 1: basic comments
                                - 2: detailed comments
    """

    fname = 'exportDeesseXInput'

    if not deesseX_input.ok:
        if verbose > 0:
            print(f'ERROR ({fname}): check deesseX input')
        return None

    # Create ouptut directory if needed
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Convert deesseX input from python to C
    try:
        mpds_xsiminput = deesseX_input_py2C(deesseX_input)
    except:
        print(f'ERROR ({fname}): unable to convert deesseX input from python to C...')
        return None

    if mpds_xsiminput is None:
        print(f'ERROR ({fname}): unable to convert deesseX input from python to C...')
        return None

    err = deesse.MPDSExportXSimInput( mpds_xsiminput, dirname, fileprefix, endofline, verbose)

    if err:
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)

    # Free memory on C side: deesseX input
    deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #deesse.MPDSFree(mpds_siminput)
    deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def importDeesseXInput(filename, dirname='.'):
    """
    Imports deesseX input from ASCII files, used for command line version of
    deesse (from the directory named <dirname>).

    :param filename:        (string) name of the general input ASCII file
                                (without path) used for the command line
                                version of deesse
    :param dirname:         (string) name of the directory in which the input
                                ASCII files are stored (and from which the
                                command line version of deesse would be
                                launched)

    :param deesseX_input:   (DeesseXInput class) deesseX input - python
    """

    fname = 'importDeesseXInput'

    # Check directory
    if not os.path.isdir(dirname):
        print(f'ERROR ({fname}): directory does not exist')
        return None

    # Check file
    if not os.path.isfile(os.path.join(dirname, filename)):
        print(f'ERROR ({fname}): input file does not exist')
        return None

    # Get current working directory
    cwd = os.getcwd()

    # Change directory
    os.chdir(dirname)

    try:
        # Initialization a double pointer onto MPDS_XSIMINPUT
        mpds_xsiminputp = deesse.new_MPDS_XSIMINPUTp()

        # Import
        deesse.MPDSImportXSimInput(filename, mpds_xsiminputp)

        # Retrieve structure
        mpds_xsiminput = deesse.MPDS_XSIMINPUTp_value(mpds_xsiminputp)

        # Convert deesse input from C to python
        deesseX_input = deesseX_input_C2py(mpds_xsiminput)

    except:
        deesseX_input = None

    if deesseX_input is None:
        print(f'ERROR ({fname}): unable to import deesseX input from ASCII files...')

    # Change directory (to initial working directory)
    os.chdir(cwd)

    return deesseX_input
# ----------------------------------------------------------------------------

##### Other classes "using deesse" #####

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
