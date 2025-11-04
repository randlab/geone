#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'deesseinterface.py'
# author:         Julien Straubhaar
# date:           jan-2018
# -------------------------------------------------------------------------

"""
Module for interfacing deesse (in C) for python.
"""

import numpy as np
import sys, os, copy
import multiprocessing

from geone import img
from geone.deesse_core import deesse
from geone.img import Img, PointSet
from geone.blockdata import BlockData

version = [deesse.MPDS_VERSION_NUMBER, deesse.MPDS_BUILD_NUMBER]

# ============================================================================
class DeesseinterfaceError(Exception):
    """
    Custom exception related to `deesseinterface` module.
    """
    pass
# ============================================================================

# ============================================================================
class SearchNeighborhoodParameters(object):
    """
    Class defining search neighborhood parameters (for deesse).

    **Attributes**

    radiusMode : str {'large_default', 'ti_range_default', 'ti_range', \
                    'ti_range_xy', 'ti_range_xz', 'ti_range_yz', 'ti_range_xyz', \
                    'manual'}, default: 'large_default'
        radius mode, defining how the search radii `rx`, `ry`, `rz` are set:

        - 'large_default': \
        large radii set according to the size of the SG and the TI(s), and \
        the use of homothethy and/or rotation for the simulation \
        (automatically computed)
        - 'ti_range_default': \
        search radii set according to the TI(s) variogram ranges, one of \
        the 5 next modes 'ti_range_*' will be used according to the use of \
        homothethy and/or rotation for the simulation \
        (automatically computed)
        - 'ti_range': \
        search radii set according to the TI(s) variogram ranges, \
        independently in each direction \
        (automatically computed)
        - 'ti_range_xy': \
        search radii set according to the TI(s) variogram ranges, \
        `rx` = `ry` independently from `rz` \
        (automatically computed)
        - 'ti_range_xz': \
        search radii set according to the TI(s) variogram ranges, \
        `rx` = `rz` independently from `ry` \
        (automatically computed)
        - 'ti_range_yz': \
        search radii set according to the TI(s) variogram ranges, \
        `ry` = `rz` independently from `rx` \
        (automatically computed)
        - 'ti_range_xyz': \
        search radii set according to the TI(s) variogram ranges, \
        `rx` = `ry` = `rz` \
        (automatically computed)
        - 'manual': \
        search radii `rx`, `ry`, `rz`, in number of cells, are \
        explicitly given

    rx : float, default: 0.0
        radius, in number of cells, along x axis direction
        (used only if radiusMode is set to 'manual')

    ry : float, default: 0.0
        radius, in number of cells, along y axis direction
        (used only if radiusMode is set to 'manual')

    rz : float, default: 0.0
        radius, in number of cells, along z axis direction
        (used only if radiusMode is set to 'manual')

    anisotropyRatioMode : str {'one', 'radius', 'radius_xy', 'radius_xz', \
                    'radius_yz', 'radius_xyz', 'manual'}, default: 'one'
        anisotropy ratio mode, defining how the anisotropy - i.e. inverse
        unit distance `ax`, `ay`, `az` along x, y, z axis - is set:

        - 'one': `ax` = `ay` = `az` = 1
        - 'radius': `ax` = `rx`, `ay` = `ry`, `az` = `rz`
        - 'radius_xy': `ax` = `ay = `max(rx, ry)`, `az` = `rz`
        - 'radius_xz': `ax` = `az = `max(rx, rz)`, `ay` = `ry`
        - 'radius_yz': `ay` = `az = `max(ry, rz)`, `ax` = `rx`
        - 'radius_xyz': `ax` = `ay` = `az` = `max(rx, ry, rz)`
        - 'manual': `ax`, `ay`, `az` explicitly given

        Notes:

        - if `anisotropyRatioMode='one'`: \
        isotropic distance - maximal distance for search neighborhood nodes \
        will be equal to the maximum of the search radii
        - if `anisotropyRatioMode='radius'`: \
        anisotropic distance - nodes at distance one on the border of the \
        search neighborhood, maximal distance for search neighborhood nodes \
        will be 1
        - if `anisotropyRatioMode='radius_*'`: \
        anisotropic distance - maximal distance for search neighborhood nodes \
        will be 1

    ax : float, default: 0.0
        anisotropy (inverse unit distance) along x axis direction

    ay : float, default: 0.0
        anisotropy (inverse unit distance) along y axis direction

    az : float, default: 0.0
        anisotropy (inverse unit distance) along z axis direction

    angle1 : float, default: 0.0
        1st angle (azimuth) in degrees for rotation

    angle2 : float, default: 0.0
        2nd angle (dip) in degrees for rotation

    angle3 : float, default: 0.0
        3rd angle (plunge) in degrees for rotation

    power : float, default: 0.0
        power for computing weight according to distance

    **Methods**
    """
    def __init__(self,
                 radiusMode='large_default',
                 rx=0.0, ry=0.0, rz=0.0,
                 anisotropyRatioMode='one',
                 ax=0.0, ay=0.0, az=0.0,
                 angle1=0.0, angle2=0.0, angle3=0.0,
                 power=0.0):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
        # fname = 'SearchNeighborhoodParameters'

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class defining probability constraints for one variable (for deesse).

    **Attributes**

    probabilityConstraintUsage : int, default: 0
        defines the usage of probability constraints:

        - 0: no probability constraint
        - 1: global probability constraints
        - 2: local probability constraints using support
        - 3: local probability constraints based on rejection

    nclass : int, default: 0
        number of classes of values;
        used if `probabilityConstraintUsage>0`

    classInterval : list of 2D array-like of floats with two columns, optional
        definition of the classes of values by intervals:

        - `classInterval[i]` : array `a` of shape (n_i, 2), defining the \
        i-th class as the union of intervals as \
        `[a[0, 0], a[0, 1][ U ... U [a[n_i-1, 0], a[n_i-1, 1][`

        used if `probabilityConstraintUsage>0`

    globalPdf : 1D array-like of floats of shape (nclass, ), optional
        global probability for each class;
        used if `probabilityConstraintUsage=1`

    localPdf : 4D array-like of floats of shape (nclass, nz, ny, nx), optional
        probability for each class:

        - `localPdf[i]` is the "map defined on the simulation grid (SG)" of \
        of dimension nx x ny x nz (number of cell along each axis)

        used if `probabilityConstraintUsage` in [2, 3]

    localPdfSupportRadius : float, default: 12.0
        support radius for local pdf;
        used if `probabilityConstraintUsage=2`

    localCurrentPdfComputation : int, default: 0
        defines the method used for computing the local current pdf:

        - 0: "COMPLETE" mode: all the informed nodes in the search \
        neighborhood, and within the support are taken into account
        - 1: "APPROXIMATE" mode: only the neighboring nodes (used for the \
        search in the TI) within the support are taken into account

        used if `probabilityConstraintUsage=2`

    comparingPdfMethod : int, default: 5
        defines the method used for comparing pdf's:

        - 0: MAE (Mean Absolute Error)
        - 1: RMSE (Root Mean Squared Error)
        - 2: KLD (Kullback Leibler Divergence)
        - 3: JSD (Jensen-Shannon Divergence)
        - 4: MLikRsym (Mean Likelihood Ratio (over each class indicator, \
        symmetric target interval))
        - 5: MLikRopt (Mean Likelihood Ratio (over each class indicator, \
        optimal target interval))

        used if `probabilityConstraintUsage` in [1, 2]

    rejectionMode : int, default: 0
        defines the mode of rejection (during the scan of the TI):

        - 0: rejection is done first (before checking pattern (and other \
        constraint)) according to acceptation probabilities proportional \
        to p[i]/q[i] (for class i), where
            - q is the marginal pdf of the scanned TI
            - p is the given local pdf at the simulated node
        - 1: rejection is done last (after checking pattern (and other \
        constraint)) according to acceptation probabilities proportional \
        to p[i] (for class i), where
            - p is the given local pdf at the simulated node

        used if `probabilityConstraintUsage=3`

    deactivationDistance : float, default: 4.0
        deactivation distance (the probability constraint is deactivated if
        the distance between the current simulated node and the last node in
        its neighbors (used for the search in the TI) (distance computed
        according to the corresponding search neighborhood parameters) is below
        the given deactivation distance);
        used if `probabilityConstraintUsage>0`

    probabilityConstraintThresholdType : int, default: 0
        defines the type of (acceptance) threhsold for pdfs' comparison:

        - 0: constant threshold
        - 1: dynamic threshold

        used if `probabilityConstraintUsage` in [1, 2]

    constantThreshold : float, default: 1.e-3
        (acceptance) threshold value for pdfs' comparison;
        used if `probabilityConstraintUsage` in [1, 2] and
        `probabilityConstraintThresholdType=0`

    dynamicThresholdParameters : sequence of 7 floats, optional
        parameters for dynamic threshold (used for pdfs' comparison);
        used if `probabilityConstraintUsage` in [1, 2] and
        `probabilityConstraintThresholdType=1`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
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
                 dynamicThresholdParameters=None,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
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
                err_msg = f'{fname}: parameter `globalPdf`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class defining connectivity constraints for one variable (for deesse).

    **Attributes**

    connectivityConstraintUsage : int, default: 0
        defines the usage of connectivity constraints:

        - 0: no connectivity constraint
        - 1: set connecting paths before the simulation by successively \
        binding the nodes to be connected in a random order
        - 2: set connecting paths before the simulation by successively \
        binding the nodes to be connected beginning with the pair of \
        nodes with the smallest distance and then the remaining nodes \
        in increasing order according to their distance to the set of \
        nodes already connected; the distance between two nodes is \
        defined as the length (in number of nodes) of the minimal path \
        binding the two nodes in an homogeneous connected medium \
        according to the type of connectivity (`connectivityType`)
        - 3: check connectivity pattern during the simulation

    connectivityType : str {'connect_face', 'connect_face_edge', \
                    'connect_face_edge_corner'}, default: 'connect_face'
        connectivity type:

        - 'connect_face': \
        6-neighbors connection (by face)
        - 'connect_face_edge': \
        18-neighbors connection (by face or edge)
        - 'connect_face_edge_corner': \
        26-neighbors connection (by face, edge or corner)

        used if `connectivityConstraintUsage>0`

    nclass : int, default: 0
        number of classes of values;
        used if `connectivityConstraintUsage>0`

    classInterval : list of 2D array-like of floats with two columns, optional
        definition of the classes of values by intervals:

        - `classInterval[i]` : array `a` of shape (n_i, 2), defining the \
        i-th class as the union of intervals as \
        `[a[0, 0], a[0, 1][ U ... U [a[n_i-1, 0], a[n_i-1, 1][`

        used if `connectivityConstraintUsage>0`

    varname : str, default: ''
        variable name for connected component label (should be in a conditioning
        data set); note: label negative or zero means no connectivity constraint

    tiAsRefFlag : bool, default: True
        - if `True`: the (first) TI is used as reference for connectivity
        - if `False`: the reference image for connectivity is given by \
        `refConnectivityImage` (possible only if `connectivityConstraintUsage=1` \
        or `connectivityConstraintUsage=2`)

        used if `connectivityConstraintUsage>0`

    refConnectivityImage : :class:`geone.img.Img`, optional
        reference image for connectivity;
        used only if `connectivityConstraintUsage` in [1, 2] and
        `tiAsRefFlag=False`

    refConnectivityVarIndex : int, default: 0
        variable index in image `refConnectivityImage` for the search of
        connected paths;
        used only if `connectivityConstraintUsage` in [1, 2] and
        `tiAsRefFlag=False`

    deactivationDistance : float, default: 0.0
        deactivation distance (the connectivity constraint is deactivated if
        the distance between the current simulated node and the last node in
        its neighbors (used for the search in the TI) (distance computed
        according to the corresponding search neighborhood parameters) is below
        the given deactivation distance);
        used if `connectivityConstraintUsage=3`

    threshold : float, default: 0.01
        threshold value for connectivity patterns comparison;
        used if `connectivityConstraintUsage=3`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
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
                 threshold=0.01,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
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
            err_msg = f'{fname}: parameter `refConnectivityImage`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.refConnectivityImage = refConnectivityImage
        self.refConnectivityVarIndex = refConnectivityVarIndex
        self.deactivationDistance = deactivationDistance
        self.threshold = threshold

    # ------------------------------------------------------------------------
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class defining "pyramid general parameters" (for all variables) (for deesse).

    **Attributes**

    npyramidLevel : int, default: 0
        number of pyramid level(s) (in addition to original simulation grid),
        integer greater than or equal to zero; if positive, pyramid is used and
        pyramid levels are indexed from fine to coarse resolution:

        - index 0            : original simulation grid
        - index npyramidLevel: coarsest level

    kx : sequence of ints of length `npyramidLevel`, optional
        reduction step along x axis for each level:

        * kx[.] = 0: nothing is done, same dimension after reduction
        * kx[.] = 1: same dimension after reduction(with weighted average over \
                     3 nodes)
        * kx[.] = 2: classical gaussian pyramid
        * kx[.] > 2: generalized gaussian pyramid


    kx : sequence of ints of length `npyramidLevel`, optional
        reduction step along y axis for each level:

        * ky[.] = 0: nothing is done, same dimension after reduction
        * ky[.] = 1: same dimension after reduction(with weighted average over \
                     3 nodes)
        * ky[.] = 2: classical gaussian pyramid
        * ky[.] > 2: generalized gaussian pyramid

    kz : sequence of ints of length `npyramidLevel`, optional
        reduction step along z axis for each level:

        * kz[.] = 0: nothing is done, same dimension after reduction
        * kz[.] = 1: same dimension after reduction(with weighted average over \
                     3 nodes)
        * kz[.] = 2: classical gaussian pyramid
        * kz[.] > 2: generalized gaussian pyramid

    pyramidSimulationMode : str {'hierarchical', 'hierarchical_using_expansion'}, \
                                default: 'hierarchical_using_expansion'
        simulation mode for pyramids:

        - 'hierarchical':
            (a) spreading conditioning data through the pyramid by simulation at \
            each level, from fine to coarse resolution, conditioned to the level \
            one rank finer
            (b) simulation at the coarsest level, then simulation of each level, \
            from coarse to fine resolution, conditioned to the level one rank \
            coarser
        - 'hierarchical_using_expansion':
            (a) spreading conditioning data through the pyramid by simulation at \
            each level, from fine to coarse resolution, conditioned to the level \
            one rank finer
            (b) simulation at the coarsest level, then simulation of each level, \
            from coarse to fine resolution, conditioned to the gaussian expansion \
            of the level one rank coarser
        - 'all_level_one_by_one':
            co-simulation of all levels, simulation done at one level at a time

    factorNneighboringNode : 1D array-like (of doubles), optional
        factors for adpating the maximal number of neighboring nodes:

        - if `pyramidSimulationMode='hierarchical'` or
            `pyramidSimulationMode='hierarchical_using_expansion'`:
            array of size `4 * npyramidLevel + 1` with entries:

            * faCond[0], faSim[0], fbCond[0], fbSim[0]
            * ...
            * faCond[n-1], faSim[n-1], fbCond[n-1], fbSim[n-1]
            * fbSim[n]

            i.e. (4*n+1) positive numbers where n = `npyramidLevel`, with the
            following meaning; the maximal number of neighboring nodes (according
            to each variable) is multiplied by

            (a) faCond[j] and faSim[j] for the conditioning level (level j) \
            and the simulated level (level j+1) resp. during step (a) above
            (b) fbCond[j] and fbSim[j] for the conditioning level (level j+1) \
            (expanded if `pyramidSimulationMode='hierarchical_using_expansion'`) \
            and the simulated level (level j) resp. during step (b) above

        - if `pyramidSimulationMode=all_level_one_by_one'`:
            array of size `npyramidLevel + 1` with entries:

            * f[0],..., f[npyramidLevel-1], f[npyramidLevel]

            i.e. `npyramidLevel + 1` positive numbers, with the following
            meaning; the maximal number of neighboring nodes (according to each
            variable) is multiplied by f[j] for the j-th pyramid level

    factorDistanceThreshold : 1D array-like of floats, optional
        factors for adpating the distance (acceptance) threshold (similar to
        `factorNneighboringNode`)

    factorMaxScanFraction : sequence of floats of length `npyramidLevel + 1`, optional
        factors for adpating the maximal scan fraction: the maximal scan
        fraction (according to each TI) is multiplied by `factorMaxScanFraction[j]`
        for the j-th pyramid level

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
    """
    def __init__(self,
                 npyramidLevel=0,
                 nx=100, ny=100, nz=100,
                 kx=None, ky=None, kz=None,
                 pyramidSimulationMode='hierarchical_using_expansion',
                 factorNneighboringNode=None,
                 factorDistanceThreshold=None,
                 factorMaxScanFraction=None,
                 logger=None):
        """
        Inits an instance of the class.

        The parameters `nx`, `ny`, `nz` are used to set default values for
        attributes `kx`, `ky`, `kz` respectively.
        As default (i.e. if `kx` is `None`): if `nx>1`, then every component
        of `kx` will be set to 2, otherwise to 0. Similarly for `ky`, `kz`.

        For other **Parameters** : see "Attributes" in the class definition above.
        """
        fname = 'PyramidGeneralParameters'

        self.npyramidLevel = npyramidLevel

        # pyramidSimulationMode
        if pyramidSimulationMode not in ('hierarchical', 'hierarchical_using_expansion', 'all_level_one_by_one'):
            err_msg = f'{fname}: unknown `pyramidSimulationMode`'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.pyramidSimulationMode = pyramidSimulationMode

        if npyramidLevel > 0:
            # kx, ky, kz
            if kx is None:
                self.kx = np.array([2 * int (nx>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kx = np.asarray(kx, dtype='int').reshape(npyramidLevel)
                except:
                    err_msg = f'{fname}: parameter `kx`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

            if ky is None:
                self.ky = np.array([2 * int (ny>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.ky = np.asarray(ky, dtype='int').reshape(npyramidLevel)
                except:
                    err_msg = f'{fname}: parameter `ky`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

            if kz is None:
                self.kz = np.array([2 * int (nz>1) for i in range(npyramidLevel)])
            else:
                try:
                    self.kz = np.asarray(kz, dtype='int').reshape(npyramidLevel)
                except:
                    err_msg = f'{fname}: parameter `kz`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

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
                        err_msg = f'{fname}: parameter `factorNneighboringNode`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        err_msg = f'{fname}: parameter `factorDistanceThreshold`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else: # pyramidSimulationMode == 'all_level_one_by_one'
                n = npyramidLevel + 1
                if factorNneighboringNode is None:
                    factorNneighboringNode = 1./n * np.ones(n)
                    self.factorNneighboringNode = factorNneighboringNode
                else:
                    try:
                        self.factorNneighboringNode = np.asarray(factorNneighboringNode, dtype=float).reshape(n)
                    except:
                        err_msg = f'{fname}: parameter `factorNneighboringNode`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

                if factorDistanceThreshold is None:
                    factorDistanceThreshold = np.ones(n)
                    self.factorDistanceThreshold = factorDistanceThreshold
                else:
                    try:
                        self.factorDistanceThreshold = np.asarray(factorDistanceThreshold, dtype=float).reshape(n)
                    except:
                        err_msg = f'{fname}: parameter `factorDistanceThreshold`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `factorMaxScanFraction`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class defining "pyramid parameters" for one variable (for deesse).

    **Attributes**

    nlevel : int, default: 0
        number of pyramid level(s) (in addition to original simulation grid)

    pyramidType : str {'none', 'continuous', 'categorical_auto', \
                    'categorical_custom', 'categorical_to_continuous'}, \
                    default: 'none'
        type of pyramid:

        - 'none': \
        no pyramid simulation
        - 'continuous': \
        pyramid applied to continuous variable (direct)
        - 'categorical_auto': \
        pyramid for categorical variable, pyramid for indicator variable of \
        each category except one (one pyramid per indicator variable)
        - 'categorical_custom': \
        pyramid for categorical variable, pyramid for indicator variable of \
        each class of values given explicitly (one pyramid per indicator \
        variable)
        - 'categorical_to_continuous': \
        pyramid for categorical variable, the variable is transformed to a \
        continuous variable (according to connection between adjacent nodes, \
        the new values are ordered such that close values correspond to the \
        most connected categories), then one pyramid for the transformed \
        variable is used

    nclass : int, default: 0
        number of classes of values;
        used if `pyramidType='categorical_custom'`

    classInterval : list of 2D array-like of floats with two columns, optional
        definition of the classes of values by intervals:

        - `classInterval[i]` : array `a` of shape (n_i, 2), defining the \
        i-th class as the union of intervals as \
        `[a[0, 0], a[0, 1][ U ... U [a[n_i-1, 0], a[n_i-1, 1][`

        used if `pyramidType='categorical_custom'`

    outputLevelFlag : sequence of bools of length `nlevel`, optional
        indicates which level is saved in output:

        - `outputLevelFlag[j]`:
            - `False`: level of index (j+1) will not be saved in output
            - `True`: level of index (j+1) will be saved in output \
            (only the pyramid for the original variables flagged \
            for output in the field `outputVarFlag` of the parent \
            class :class:`DeesseInput` will be saved);

        Notes:

        - the name of the output variables are set to \
        '<vname>_ind<i>_lev<k>_real<n>' where
            - <vname> is the name of the "original" variable,
            - <i> is a pyramid index for that variable which starts at 0 (more \
            than one index can be required if the pyramid type is set to \
            'categorical_auto' or 'categorical_custom')
            - <k> is the level index
            - <n> is the realization index (starting from 0)
        - the values of the output variables are the normalized values (as used \
        during the simulation in every level)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
    """
    def __init__(self,
                 nlevel=0,
                 pyramidType='none',
                 nclass=0,
                 classInterval=None,
                 outputLevelFlag=None,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
        fname = 'PyramidParameters'

        self.nlevel = nlevel

        if pyramidType not in ('none', 'continuous', 'categorical_auto', 'categorical_custom', 'categorical_to_continuous'):
            err_msg = f'{fname}: unknown `pyramidType`'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.pyramidType = pyramidType

        self.nclass = nclass
        self.classInterval = classInterval

        if outputLevelFlag is None:
            self.outputLevelFlag = np.array([False for i in range(nlevel)], dtype='bool') # set dtype='bool' in case of nlevel=0
        else:
            try:
                self.outputLevelFlag = np.asarray(outputLevelFlag, dtype='bool').reshape(nlevel)
            except:
                err_msg = f'{fname}: parameter `outputLevelFlag`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

    # ------------------------------------------------------------------------
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class defining main input parameters for deesse.

    **Attributes**

    simName : str, default: 'deesse_py'
        simulation name (useless)

    nx : int, default: 1
        number of simulation grid (SG) cells along x axis

    ny : int, default: 1
        number of simulation grid (SG) cells along y axis

    nz : int, default: 1
        number of simulation grid (SG) cells along z axis

        Note: `(nx, ny, nz)` is the SG dimension (in number of cells)

    sx : float, default: 1.0
        cell size along x axis

    sy : float, default: 1.0
        cell size along y axis

    sz : float, default: 1.0
        cell size along z axis

        Note: `(sx, sy, sz)` is the cell size in SG

    ox : float, default: 0.0
        origin of the simulation grid (SG) along x axis (x coordinate of cell border)

    oy : float, default: 0.0
        origin of the simulation grid (SG) along y axis (y coordinate of cell border)

    oz : float, default: 0.0
        origin of the simulation grid (SG) along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the SG

    nv : int, default: 0
        number of variable(s) / attribute(s), should be at least 1 to launch
        deesse

    varname : sequence of strs of length `nv`, optional
        variable names

    outputVarFlag : sequence of bools of length `nv`, optional
        flags indicating which variable(s) is (are) saved in output

    outputPathIndexFlag : bool, default: False
        indicates if path index maps are retrieved in output;
        path index map: index in the simulation path

    outputErrorFlag : bool, default: False
        indicates if error maps are retrieved in output;
        error map: error for the retained candidate

    outputTiGridNodeIndexFlag : bool, default: False
        indicates if TI grid node index maps are retrieved in output;
        TI grid node index map: index of the grid node of the retained candidate
        in the TI

    outputTiIndexFlag : bool, default: False
        indicates if TI index maps are retrieved in output;
        TI index map: index of the TI used (makes sense if number of TIs (nTI)
        is greater than 1)

    outputReportFlag : bool, default: False
        indicates if a report file will be written

    outputReportFileName : str, optional
        name of the report file (if `outputReportFlag=True`)

    nTI : int, optional
        number of training image(s) (TI(s))
        (deprecated, computed automatically from `TI` and `simGridAsTiFlag`,
        `nTI=None` should be used)

    TI : [sequence of] :class:`geone.img.Img`
        training image(s) (TI(s)) used for the simulation, may contain `None`
        entries; it must be compatible with `simGridAsTiFlag`

    simGridAsTiFlag : [sequence of] bool(s), optional
        flags indicating if the simulation grid itself is used as TI, for each
        TI; by default (`None`): an array of `False` is considered

    pdfTI : array-like of floats, optional
        array of shape (nTI, nz, ny, nx) (reshaped if needed), probability for
        TI selection:

        - `pdfTI[i]` is the "map defined on the SG" of the probability to \
        select the i-th TI

        used if more than one TI are used (`nTI>1`)

    dataImage : sequence of :class:`geone.img.Img`, optional
        list of data image(s); image(s) used as conditioning data, each data
        image should have the same grid dimensions as those of the SG and its
        variable name(s) should be included in the list `varname`

    dataPointSet : sequence of :class:`geone.img.PointSet`, optional
        list of data point set(s); point set(s) used as conditioning data,
        each data point set should have at least 4 variables: 'X', 'Y', 'Z',
        the coordinates in the SG and at least one variable with name in the
        list `varname`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    homothetyUsage : int, default: 0
        defines the usage of homothety:

        - 0: no homothety
        - 1: homothety without tolerance
        - 2: homothety with tolerance

    homothetyXLocal : bool, default: False
        indicates if homothety according to X axis is local (`True`) or global
        (`False`);
        used if `homothetyUsage>0`

    homothetyXRatio : array-like of floats, or float, optional
        homothety ratio according to X axis:

        -if `homothetyUsage=1`:
            - if `homothetyXLocal=True`: \
            3D array of shape (nz, ny, nx): values in the SG
            - else: \
            float: value
        -if `homothetyUsage=2`:
            - if `homothetyXLocal=True`: \
            4D array of shape (2, nz, ny, nx): \
                min values (`homothetyXRatio[0]`) and \
                max values (`homothetyXRatio[1]`) in the SG
            - else: \
            sequence of 2 floats: min value and max value

        used if `homothetyUsage>0`

    homothetyYLocal :
        as `homothetyXLocal`, but for the Y axis

    homothetyYRatio :
        as `homothetyXRatio`, but for the Y axis

    homothetyZLocal :
        as `homothetyXLocal`, but for the Z axis

    homothetyZRatio :
        as `homothetyXRatio`, but for the Z axis

    rotationUsage : int, default: 0
        defines the usage of rotation:

        - 0: no rotation
        - 1: rotation without tolerance
        - 2: rotation with tolerance

    rotationAzimuthLocal : bool, default: False
        indicates if azimuth angle is local (`True`) or global (`False`);
        used if `rotationUsage>0`

    rotationAzimuth : array-like of floats, or float, optional
        azimuth angle in degrees:

        -if `rotationUsage=1`:
            - if `rotationAzimuth=True`: \
            3D array of shape (nz, ny, nx): values in the SG
            - else: \
            float: value
        -if `rotationUsage=2`:
            - if `rotationAzimuth=True`: \
            4D array of shape (2, nz, ny, nx): \
                min values (`rotationAzimuth[0]`) and \
                max values (`rotationAzimuth[1]`) in the SG
            - else: \
            sequence of 2 floats: min value and max value

        used if `rotationUsage>0`

    rotationDipLocal :
        as `rotationAzimuthLocal`, but for the dip angle

    rotationDip :
        as `rotationAzimuth`, but for the dip angle

    rotationPlungeLocal :
        as `rotationAzimuthLocal`, but for the plunge angle

    rotationPlunge :
        as `rotationAzimuth`, but for the plunge angle

    expMax : float, default: 0.05
        maximal expansion (negative to not check consistency); the following is
        applied for each variable separetely:

        - for variable with distance type set to 0 (see below):
            * expMax >= 0: \
            if a conditioning data value is not in the set of TI values, \
            an error occurs
            * expMax < 0: \
            if a conditioning data value is not in the set of TI values, \
            a warning is displayed (no error occurs)
        - for variable with distance type not set to 0 (see below): \
        if relative distance flag is set to 1 (see below), nothing is done, \
        else:
            * expMax >= 0: \
            maximal accepted expansion of the range of the TI values for \
            covering the conditioning data values:
                - if conditioning data values are within the range of the TI \
                values: nothing is done
                - if a conditioning data value is out of the range of the TI \
                values: let
                    * new_min_ti = min ( min_cd, min_ti )
                    * new_max_ti = max ( max_cd, max_ti )

                with
                    * min_cd, max_cd, the min and max of the conditioning values
                    * min_ti, max_ti, the min and max of the TI values

                if new_max_ti-new_min_ti <= (1+expMax)*(ti_max-ti_min), then \
                the TI values are linearly rescaled from [ti_min, ti_max] to \
                [new_ti_min, new_ti_max], and a warning is displayed (no error \
                occurs); otherwise, an error occurs.
            * expMax < 0: \
            if a conditioning data value is out of the range of the TI \
            values, a warning is displayed (no error occurs), the TI values \
            are not modified

    normalizingType : str {'linear', 'uniform', 'normal'}, default: 'linear'
        normalizing type for continuous variable(s) (with distance type not
        equal to 0)

    searchNeighborhoodParameters : [sequence of] :class:`SearchNeighborhoodParameters`, optional
        search neighborhood parameters for each variable
        (sequence of length `nv`)

    nneighboringNode : [sequence of] int(s), optional
        maximal number of neighbors in the search pattern, for each variable
        (sequence of length `nv`)

    maxPropInequalityNode : [sequence of] double(s), optional
        maximal proportion of nodes with inequality data in the search pattern,
        for each variable
        (sequence of length `nv`)

    neighboringNodeDensity : [sequence of] double(s), optional
        density of neighbors in the search pattern, for each variable
        (sequence of length `nv`)

    rescalingMode : [sequence of] str(s), optional
        rescaling mode for each variable, {'none', 'min_max', 'mean_length'}
        (sequence of length `nv`)

    rescalingTargetMin : [sequence of] double(s), optional
        target min value, for each variable (used for variable with
        `rescalingMode` set to 'min_max')
        (sequence of length `nv`)

    rescalingTargetMax : [sequence of] double(s), optional
        target max value, for each variable (used for variable with
        `rescalingMode` set to 'min_max')
        (sequence of length `nv`)

    rescalingTargetMean : [sequence of] double(s), optional
        target mean value, for each variable (used for variable with
        `rescalingMode` set to 'mean_length')
        (sequence of length `nv`)

    rescalingTargetLength : [sequence of] double(s), optional
        target length value, for each variable (used for variable with
        `rescalingMode` set to 'mean_length')
        (sequence of length `nv`)

    relativeDistanceFlag : [sequence of] bool(s), optional
        flag indicating if relative distance is used (True) or not (False),
        for each variable
        (sequence of length `nv`)

    distanceType : [sequence of] int(s) or str(s), optional
        type of distance (between pattern) for each variable

        - 0 or 'categorical' : non-matching nodes (default if None)
        - 1 or 'continuous'  : L-1 distance
        - 2 : L-2 distance
        - 3 : L-p distance, requires the parameter p (positive float)
        - 4 : L-infinity

        (sequence of length `nv`)

    powerLpDistance : [sequence of] double(s), optional
        p parameter for L-p distance, for each variable (used for variable using
        L-p distance)
        (sequence of length `nv`)

    powerLpDistanceInv : [sequence of] double(s), optional
        p parameter for L-p distance, for each variable (used for variable using
        L-p distance)
        (sequence of length `nv`)

    conditioningWeightFactor : [sequence of] float(s), optional
        weight factor for conditioning data, for each variable
        (sequence of length `nv`)

    simType : str {'sim_one_by_one', 'sim_variable_vector'}, default: 'sim_one_by_one'
        simulation type:

        - 'sim_one_by_one': successive simulation of one variable \
        at one node in the simulation grid (4D path)
        - 'sim_variable_vector': successive simulation of all variable(s) \
        at one node in the simulation grid (3D path)

    simPathType : str {'random', \
                    'random_hd_distance_pdf', 'random_hd_distance_sort', \
                    'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort', \
                    'unilateral'}, default: 'random'
        simulation path type:

        - 'random': random path
        - 'random_hd_distance_pdf': random path set according to \
        distance to conditioning nodes based on pdf, \
        requires parameter `simPathStrength`
        - 'random_hd_distance_sort': random path set according to \
        distance to conditioning nodes based on sort (with a \
        random noise contribution), \
        requires parameter `simPathStrength`
        - 'random_hd_distance_sum_pdf': random path set according to \
        sum of distance to conditioning nodes based on pdf, \
        requires parameters `simPathPower` and `simPathStrength`
        - 'random_hd_distance_sum_sort': random path set according to \
        sum of distance to conditioning nodes based on sort (with \
        a random noise contribution), \
        required fields 'simPathPower' and 'simPathStrength'
        - 'unilateral': unilateral path or stratified random path, \
        requires parameter `simPathUnilateralOrder`

    simPathStrength : double, optional
        strength in [0,1] attached to distance, if `simPathType` in
        ('random_hd_distance_pdf', 'random_hd_distance_sort',
        'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort')

    simPathPower : double, optional
        power (>0) to which the distance to each conditioning node are raised,
        if `simPathType` is in
        ('random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort')

    simPathUnilateralOrder : sequence of ints, optional
        used if `simPathType='unilateral'`:

        - if `simType='sim_one_by_one'`: `simPathUnilateralOrder` is of \
        length 4, example: [0, -2, 1, 0] means that the path will visit \
        all nodes: randomly in xv-sections, with increasing z-coordinate, \
        and then decreasing y-coordinate
        - if `simType='sim_variable_vector'`: `simPathUnilateralOrder` is of \
        length 3, example: [-1, 0, 2] means that the path will visit \
        all nodes: randomly in y-sections, with decreasing x-coordinate, \
        and then increasing z-coordinate

    distanceThreshold : [sequence of] float(s), optional
        distance (acceptance) for each variable
        (sequence of length `nv`)

    softProbability : [sequence of] :class:`SoftProbability`, optional
        probability constraints parameters for each variable
        (sequence of length `nv`)

    connectivity : [sequence of] :class:`SoftProbability`, optional
        connectivity constraints parameters for each variable
        (sequence of length `nv`)

    blockData : [sequence of] :class:`geone.blockdata.BlockData`, optional
        block data parameters for each variable
        (sequence of length `nv`)

    maxScanFraction : [sequence of] double(s), optional
        maximal scan fraction of each TI
        (sequence of length `nTI`)

    pyramidGeneralParameters : :class:`PyramidGeneralParameters`, optional
        general pyramid parameters

    pyramidParameters : [sequence of] :class:`PyramidParameters`, optional
        pyramid parameters for each variable
        (sequence of length `nv`)

    pyramidDataImage : sequence of :class:`geone.img.Img`, optional
        list of data image(s); image(s) used as conditioning data in pyramid
        (in additional levels); for each image:

        - the variables are identified by their name: \
        the name should be set to '<vname>_ind<j>_lev<k>', where
            - <vname> is the name of the "original" variable,
            - <j> is the pyramid index for that variable, and
            - <k> is the level index in {1, ...} \
            (<j> and <k> are written on 3 digits with leading zeros)
        - the conditioning data values are the (already) normalized \
        values (as used during the simulation in every level)
        - the grid dimensions (support) of the level in which the data \
        are given are used: the image grid must be compatible

        Note: conditioning data integrated in pyramid may erased (replaced)
        data already set or computed from conditioning data at the level one
        rank finer

    pyramidDataPointSet : sequence of :class:`geone.img.PointSet`, optional
        list of data point set(s); point set(s) used as conditioning data in
        pyramid (in additional levels); for each point set:

        - the variables are identified by their name: \
        the name should be set to '<vname>_ind<j>_lev<k>', where
            - <vname> is the name of the "original" variable,
            - <j> is the pyramid index for that variable, and
            - <k> is the level index in {1, ...} \
            (<j> and <k> are written on 3 digits with leading zeros)
        - the conditioning data values are the (already) normalized \
        values (as used during the simulation in every level)
        - the grid dimensions (support) of the level in which the data \
        are given are used: locations (coordinates) of the points \
        must be given accordingly

        Note: conditioning data integrated in pyramid may erased (replaced)
        data already set or computed from conditioning data at the level one
        rank finer

    tolerance : float, default: 0.0
        tolerance on the (acceptance) threshold value for flagging nodes
        (for post-processing)

    npostProcessingPathMax : int, default: 0
        maximal number of post-processing path(s) (0 for no post-processing)

    postProcessingNneighboringNode : [sequence of] int(s), optional
        maximal number of neighbors in the search pattern, for each variable,
        for all post-processing paths
        (sequence of length `nv`)

    postProcessingNeighboringNodeDensity : [sequence of] double(s), optional
        density of neighbors in the search pattern, for each variable,
        for all post-processing paths
        (sequence of length `nv`)

    postProcessingDistanceThreshold : [sequence of] float(s), optional
        distance (acceptance) for each variable,
        for all post-processing paths
        (sequence of length `nv`)

    postProcessingMaxScanFraction : [sequence of] double(s), optional
        maximal scan fraction of each TI,
        for all post-processing paths
        (sequence of length `nTI`)

    postProcessingTolerance : float, default: 0.0
        tolerance on the (acceptance) threshold value for flagging nodes
        (for post-processing),
        for all post-processing paths

    seed : int, default: 1234
        initial seed

    seedIncrement : int, default: 1
        seed increment

    nrealization : int, default: 1
        number of realization(s)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    In output simulated images (obtained by running DeeSse), the names of the
    output variables are set to '<vname>_real<n>', where

    - <vname> is the name of the variable,
    - <n> is the realization index (starting from 0) \
    [<n> is written on 5 digits, with leading zeros]

    **Methods**
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
                 nrealization=1,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
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
            self.varname = ['V{i:d}' for i in range(nv)]
        else:
            try:
                self.varname = list(np.asarray(varname).reshape(nv))
            except:
                err_msg = f'{fname}: parameter `varname`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # dimension
        dim = int(nx>1) + int(ny>1) + int(nz>1)

        # outputVarFlag
        if outputVarFlag is None:
            self.outputVarFlag = np.array([True for i in range(nv)], dtype='bool')
        else:
            try:
                self.outputVarFlag = np.asarray(outputVarFlag, dtype='bool').reshape(nv)
            except:
                err_msg = f'{fname}: parameter `outputVarFlag`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
            err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid (both `None`)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if TI is not None:
            self.TI = np.asarray(TI).reshape(-1)

        if simGridAsTiFlag is not None:
            self.simGridAsTiFlag = np.asarray(simGridAsTiFlag, dtype='bool').reshape(-1)

        if TI is None:
            self.TI = np.array([None for i in range(len(self.simGridAsTiFlag))], dtype=object)

        if simGridAsTiFlag is None:
            self.simGridAsTiFlag = np.array([False for i in range(len(self.TI))], dtype='bool') # set dtype='bool' in case of len(self.TI)=0

        if len(self.TI) != len(self.simGridAsTiFlag):
            err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid (not same length)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        for f, t in zip(self.simGridAsTiFlag, self.TI):
            if (not f and t is None) or (f and t is not None):
                err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if nTI is not None and nTI != len(self.TI):
            err_msg = f'{fname}: `nTI` invalid...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `pdfTI`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `mask`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # homothety
        if homothetyUsage == 1:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif homothetyUsage == 2:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif homothetyUsage == 0:
            self.homothetyXRatio = None
            self.homothetyYRatio = None
            self.homothetyZRatio = None

        else:
            err_msg = f'{fname}: `homothetyUsage` invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif rotationUsage == 2:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0., 0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0., 0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0., 0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif rotationUsage == 0:
            self.rotationAzimuth = None
            self.rotationDip = None
            self.rotationPlunge = None

        else:
            err_msg = f'{fname}: `rotationUsage` invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `searchNeighborhoodParameters`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `nneighboringNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if maxPropInequalityNode is None:
            self.maxPropInequalityNode = np.array([0.25 for i in range(nv)])
        else:
            try:
                self.maxPropInequalityNode = np.asarray(maxPropInequalityNode).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `maxPropInequalityNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if neighboringNodeDensity is None:
            self.neighboringNodeDensity = np.array([1. for i in range(nv)])
        else:
            try:
                self.neighboringNodeDensity = np.asarray(neighboringNodeDensity, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `neighboringNodeDensity`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # rescaling
        if rescalingMode is None:
            self.rescalingMode = ['none' for i in range(nv)]
        else:
            try:
                self.rescalingMode = list(np.asarray(rescalingMode).reshape(nv))
            except:
                err_msg = f'{fname}: parameter `rescalingMode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMin is None:
            self.rescalingTargetMin = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMin = np.asarray(rescalingTargetMin, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMin`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMax is None:
            self.rescalingTargetMax = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMax = np.asarray(rescalingTargetMax, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMax`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMean is None:
            self.rescalingTargetMean = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMean = np.asarray(rescalingTargetMean, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMean`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetLength is None:
            self.rescalingTargetLength = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetLength = np.asarray(rescalingTargetLength, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetLength`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # distance, ...
        if relativeDistanceFlag is None:
            self.relativeDistanceFlag = np.array([False for i in range(nv)], dtype='bool') # set dtype='bool' in case of nv=0
        else:
            try:
                self.relativeDistanceFlag = np.asarray(relativeDistanceFlag, dtype='bool').reshape(nv)
            except:
                err_msg = f'{fname}: parameter `relativeDistanceFlag`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if powerLpDistance is None:
            self.powerLpDistance = np.array([1. for i in range(nv)])
        else:
            try:
                self.powerLpDistance = np.asarray(powerLpDistance, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `powerLpDistance`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                            err_msg = f'{fname}: parameter `distanceType`...'
                            if logger: logger.error(err_msg)
                            raise DeesseinterfaceError(err_msg)

                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `distanceType`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # conditioning weight
        if conditioningWeightFactor is None:
            self.conditioningWeightFactor = np.array([1. for i in range(nv)])
        else:
            try:
                self.conditioningWeightFactor = np.asarray(conditioningWeightFactor, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `conditioningWeightFactor`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # simulation type and simulation path type
        if simType not in ('sim_one_by_one', 'sim_variable_vector'):
            err_msg = f'{fname}: parameter `simType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.simType = simType

        if simPathType not in (
                'random',
                'random_hd_distance_pdf', 'random_hd_distance_sort',
                'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort',
                'unilateral'):
            err_msg = f'{fname}: parameter `simPathType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `simPathUnilateralOrder`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

        else:
            self.simPathUnilateralOrder = None

        # distance threshold
        if distanceThreshold is None:
            self.distanceThreshold = np.array([0.05 for i in range(nv)])
        else:
            try:
                self.distanceThreshold = np.asarray(distanceThreshold, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `distanceThreshold`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # soft probability
        if softProbability is None:
            self.softProbability = np.array([SoftProbability(probabilityConstraintUsage=0, logger=logger) for i in range(nv)])
        else:
            try:
                self.softProbability = np.asarray(softProbability).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `softProbability`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # connectivity
        if connectivity is None:
            self.connectivity = np.array([Connectivity(connectivityConstraintUsage=0, logger=logger) for i in range(nv)])
        else:
            try:
                self.connectivity = np.asarray(connectivity).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `connectivity`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # block data
        if blockData is None:
            self.blockData = np.array([BlockData(blockDataUsage=0) for i in range(nv)])
        else:
            try:
                self.blockData = np.asarray(blockData).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `blockData`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `maxScanFraction`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # pyramids
        if pyramidGeneralParameters is None:
            self.pyramidGeneralParameters = PyramidGeneralParameters(nx=nx, ny=ny, nz=nz, logger=logger)
        else:
            self.pyramidGeneralParameters = pyramidGeneralParameters

        if pyramidParameters is None:
            self.pyramidParameters = np.array([PyramidParameters() for _ in range(nv)])
        else:
            try:
                self.pyramidParameters = np.asarray(pyramidParameters).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `pyramidParameters`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingNneighboringNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingNeighboringNodeDensity`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingDistanceThreshold`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if postProcessingMaxScanFraction is None:
            self.postProcessingMaxScanFraction = np.array([min(deesse.MPDS_POST_PROCESSING_MAX_SCAN_FRACTION_DEFAULT, self.maxScanFraction[i]) for i in range(nTI)], dtype=float)

        else:
            try:
                self.postProcessingMaxScanFraction = np.asarray(postProcessingMaxScanFraction, dtype=float).reshape(nTI)
            except:
                err_msg = f'{fname}: parameter `postProcessingMaxScanFraction`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** DeesseInput object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
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

    im_c = deesse.malloc_MPDS_IMAGE()
    deesse.MPDSInitImage(im_c)

    err = deesse.MPDSMallocImage(im_c, im_py.nxyz(), im_py.nv)
    if err:
        # Free memory on C side
        deesse.MPDSFreeImage(im_c)
        deesse.free_MPDS_IMAGE(im_c)
        # Raise error
        err_msg = f'{fname}: cannot convert image from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        # deesse.charp_array_setitem(im_c.varName, i, im_py.varname[i]) # does not work!

    v = im_py.val.reshape(-1)
    np.putmask(v, np.isnan(v), deesse.MPDS_MISSING_VALUE)
    deesse.mpds_set_real_vector_from_array(im_c.var, 0, v)
    np.putmask(v, v == deesse.MPDS_MISSING_VALUE, np.nan) # replace missing_value by np.nan (restore) (v is not a copy...)

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

    varname = [deesse.mpds_get_varname(im_c.varName, i) for i in range(im_c.nvar)]
    # varname = [deesse.charp_array_getitem(im_c.varName, i) for i in range(im_c.nvar)] # also works

    v = np.zeros(nxyzv)
    deesse.mpds_get_array_from_real_vector(im_c.var, 0, v)

    im_py = Img(nx=im_c.grid.nx, ny=im_c.grid.ny, nz=im_c.grid.nz,
                sx=im_c.grid.sx, sy=im_c.grid.sy, sz=im_c.grid.sz,
                ox=im_c.grid.ox, oy=im_c.grid.oy, oz=im_c.grid.oz,
                nv=im_c.nvar, val=v, varname=varname,
                logger=logger)

    np.putmask(im_py.val, im_py.val == deesse.MPDS_MISSING_VALUE, np.nan)

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
        raise DeesseinterfaceError(err_msg)

    nvar = ps_py.nv - 3

    ps_c = deesse.malloc_MPDS_POINTSET()
    deesse.MPDSInitPointSet(ps_c)

    err = deesse.MPDSMallocPointSet(ps_c, ps_py.npt, nvar)
    if err:
        # Free memory on C side
        deesse.MPDSFreePointSet(ps_c)
        deesse.free_MPDS_POINTSET(ps_c)
        # Raise error
        err_msg = f'{fname}: cannot convert point set from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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

    varname = ['X', 'Y', 'Z'] + [deesse.mpds_get_varname(ps_c.varName, i) for i in range(ps_c.nvar)]

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

    ps_py = PointSet(npt=ps_c.npoint,
                     nv=ps_c.nvar+3, val=v, varname=varname)

    np.putmask(ps_py.val, ps_py.val == deesse.MPDS_MISSING_VALUE, np.nan)

    return ps_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def classInterval2classOfValues(classInterval):
    """
    Converts classInterval (python) to classOfValues (C).

    Parameters
    ----------
    classInterval : list of 2D array-like of floats with two columns
        definition of the classes of values by intervals:

        - `classInterval[i]` : array `a` of shape (n_i, 2), defining the \
        i-th class as the union of intervals as \
        `[a[0, 0], a[0, 1][ U ... U [a[n_i-1, 0], a[n_i-1, 1][`

    Returns
    -------
    cv : \\(MPDS_CLASSOFVALUES \\*\\)
        classOfValues (C)
    """
    # fname = 'classInterval2classOfValues'

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

    Parameters
    ----------
    cv : \\(MPDS_CLASSOFVALUES \\*\\)
        classOfValues (C)

    Returns
    -------
    classInterval : list of 2D array-like of floats with two columns
        definition of the classes of values by intervals:

        - `classInterval[i]` : array `a` of shape (n_i, 2), defining the \
        i-th class as the union of intervals as \
        `[a[0, 0], a[0, 1][ U ... U [a[n_i-1, 0], a[n_i-1, 1][`
    """
    # fname = 'classOfValues2classInterval'

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
def search_neighborhood_parameters_py2C(sn_py, logger=None):
    """
    Converts search neighborhood parameters from python to C.

    Parameters
    ----------
    sn_py : :class:`SearchNeighborhoodParameters`
        search neighborhood parameters in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sn_c : \\(MPDS_SEARCHNEIGHBORHOODPARAMETERS \\*\\)
        search neighborhood parameters in C
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
        sn_c.radiusMode = radiusMode_dict[sn_py.radiusMode]
    except:
        # Free memory on C side
        deesse.MPDSFreeSearchNeighborhoodParameters(sn_c)
        deesse.free_MPDS_SEARCHNEIGHBORHOODPARAMETERS(sn_c)
        err_msg = f'{fname}: radius mode (search neighborhood parameters) unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    sn_c.rx = sn_py.rx
    sn_c.ry = sn_py.ry
    sn_c.rz = sn_py.rz

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
        sn_c.anisotropyRatioMode = anisotropyRatioMode_dict[sn_py.anisotropyRatioMode]
    except:
        # Free memory on C side
        deesse.MPDSFreeSearchNeighborhoodParameters(sn_c)
        deesse.free_MPDS_SEARCHNEIGHBORHOODPARAMETERS(sn_c)
        err_msg = f'{fname}: anisotropy ratio mode (search neighborhood parameters) unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    sn_c.ax = sn_py.ax
    sn_c.ay = sn_py.ay
    sn_c.az = sn_py.az
    sn_c.angle1 = sn_py.angle1
    sn_c.angle2 = sn_py.angle2
    sn_c.angle3 = sn_py.angle3
    if sn_c.angle1 != 0 or sn_c.angle2 != 0 or sn_c.angle3 != 0:
        sn_c.rotationFlag = deesse.TRUE
    else:
        sn_c.rotationFlag = deesse.FALSE
    sn_c.power = sn_py.power

    return sn_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def search_neighborhood_parameters_C2py(sn_c, logger=None):
    """
    Converts search neighborhood parameters from C to python.

    Parameters
    ----------
    sn_c : \\(MPDS_SEARCHNEIGHBORHOODPARAMETERS \\*\\)
        search neighborhood parameters in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sn_py : :class:`SearchNeighborhoodParameters`
        search neighborhood parameters in python
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
        err_msg = f'{fname}: radius mode (search neighborhood parameters) unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        err_msg = f'{fname}: anisotropy ratio mode (search neighborhood parameters) unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    ax = sn_c.ax
    ay = sn_c.ay
    az = sn_c.az
    angle1 = sn_c.angle1
    angle2 = sn_c.angle2
    angle3 = sn_c.angle3
    power = sn_c.power

    sn_py = SearchNeighborhoodParameters(
        radiusMode=radiusMode,
        rx=rx, ry=ry, rz=rz,
        anisotropyRatioMode=anisotropyRatioMode,
        ax=ax, ay=ay, az=az,
        angle1=angle1, angle2=angle2, angle3=angle3,
        power=power
    )

    return sn_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def set_simAndPathParameters_C(
        simType,
        simPathType,
        simPathStrength,
        simPathPower,
        simPathUnilateralOrder,
        logger=None):
    """
    Sets simAndPathParameters (C) from relevant parameters (python).

    Parameters
    ----------
    simType : str
        simulation type

    simPathType : str
        simulation path type

    simPathStrength : double, or None
        strength in [0,1] attached to distance

    simPathPower : double, or None
        power (>0) to which the distance to each conditioning node are raised

    simPathUnilateralOrder : sequence of ints, or None
        defines unilatera path

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sapp_c : \\(MPDS_SIMANDPATHPARAMETERS \\*\\)
        simAndPathParameters (C)
    """
    fname = 'set_simAndPathParameters_C'

    sapp_c = deesse.malloc_MPDS_SIMANDPATHPARAMETERS()
    deesse.MPDSInitSimAndPathParameters(sapp_c)

    if simType == 'sim_one_by_one':
        sapp_c.simType = deesse.SIM_ONE_BY_ONE
    elif simType == 'sim_variable_vector':
        sapp_c.simType = deesse.SIM_VARIABLE_VECTOR
    else:
        # Free memory on C side
        deesse.MPDSFreeSimAndPathParameters(sapp_c)
        deesse.free_MPDS_SIMANDPATHPARAMETERS(sapp_c)
        err_msg = f'{fname}: simulation type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        # Free memory on C side
        deesse.MPDSFreeSimAndPathParameters(sapp_c)
        deesse.free_MPDS_SIMANDPATHPARAMETERS(sapp_c)
        err_msg = f'{fname}: simulation path type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    return sapp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def softProbability_py2C(
        sp_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts soft probability parameters from python to C.

    Parameters
    ----------
    sp_py : :class:`SoftProbability`
        soft probability parameters in python

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

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sp_c : \\(MPDS_SOFTPROBABILITY \\*\\)
        soft probability parameters in C
    """
    fname = 'softProbability_py2C'

    sp_c = deesse.malloc_MPDS_SOFTPROBABILITY()
    deesse.MPDSInitSoftProbability(sp_c)

    # ... probabilityConstraintUsage
    sp_c.probabilityConstraintUsage = sp_py.probabilityConstraintUsage
    if sp_py.probabilityConstraintUsage == 0:
        return sp_c

    # ... classOfValues
    sp_c.classOfValues = classInterval2classOfValues(sp_py.classInterval)

    if sp_py.probabilityConstraintUsage == 1:
        # ... globalPdf
        sp_c.globalPdf = deesse.new_real_array(sp_py.nclass)
        deesse.mpds_set_real_vector_from_array(
            sp_c.globalPdf, 0,
            np.asarray(sp_py.globalPdf).reshape(sp_py.nclass))

    elif sp_py.probabilityConstraintUsage == 2 or sp_py.probabilityConstraintUsage == 3:
        # ... localPdf
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=sp_py.nclass, val=sp_py.localPdf,
                 logger=logger)
        try:
            sp_c.localPdfImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSoftProbability(sp_c)
            deesse.free_MPDS_SOFTPROBABILITY(sp_c)
            err_msg = f'{fname}: cannot convert local pdf image from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

    if sp_py.probabilityConstraintUsage == 2:
        # ... localPdfSupportRadius
        sp_c.localPdfSupportRadius = deesse.new_real_array(1)
        deesse.mpds_set_real_vector_from_array(
            sp_c.localPdfSupportRadius, 0,
            np.asarray(sp_py.localPdfSupportRadius).reshape(1))

        # ... localCurrentPdfComputation
        sp_c.localCurrentPdfComputation = sp_py.localCurrentPdfComputation

    if sp_py.probabilityConstraintUsage == 1 or sp_py.probabilityConstraintUsage == 2:
        # ... comparingPdfMethod
        sp_c.comparingPdfMethod = sp_py.comparingPdfMethod

        # ... probabilityConstraintThresholdType
        sp_c.probabilityConstraintThresholdType = sp_py.probabilityConstraintThresholdType

        # ... constantThreshold
        sp_c.constantThreshold = sp_py.constantThreshold

        if sp_py.probabilityConstraintThresholdType == 1:
            # ... dynamicThresholdParameters
            sp_c.dynamicThresholdParameters = deesse.new_real_array(7)
            deesse.mpds_set_real_vector_from_array(
                sp_c.dynamicThresholdParameters, 0,
                np.asarray(sp_py.dynamicThresholdParameters).reshape(7))

    if sp_py.probabilityConstraintUsage == 3:
        # ... rejectionMode
        sp_c.rejectionMode = sp_py.rejectionMode

    # ... deactivationDistance
    sp_c.deactivationDistance = sp_py.deactivationDistance

    return sp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def softProbability_C2py(sp_c, logger=None):
    """
    Converts soft probability parameters from C to python.

    Parameters
    ----------
    sp_c : \\(MPDS_SOFTPROBABILITY \\*\\)
        soft probability parameters in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sp_py : :class:`SoftProbability`
        soft probability parameters in python
    """
    # fname = 'softProbability_C2py'

    # ... probabilityConstraintUsage
    probabilityConstraintUsage = sp_c.probabilityConstraintUsage
    if probabilityConstraintUsage == 0:
        sp_py = SoftProbability()
        return sp_py

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
        im = img_C2py(sp_c.localPdfImage, logger=logger)
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

    sp_py = SoftProbability(
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
        dynamicThresholdParameters=dynamicThresholdParameters,
        logger=logger
    )

    return sp_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def connectivity_py2C(co_py, logger=None):
    """
    Converts connectivity parameters from python to C.

    Parameters
    ----------
    co_py : :class:`Connectivity`
        connectivity parameters in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    co_c : \\(MPDS_CONNECTIVITY \\*\\)
        connectivity parameters in C
    """
    fname = 'connectivity_py2C'

    co_c = deesse.malloc_MPDS_CONNECTIVITY()
    deesse.MPDSInitConnectivity(co_c)

    # ... connectivityConstraintUsage
    co_c.connectivityConstraintUsage = co_py.connectivityConstraintUsage
    if co_py.connectivityConstraintUsage == 0:
        return co_c

    # ... connectivityType
    connectivityType_dict = {
        'connect_face'             : deesse.CONNECT_FACE,
        'connect_face_edge'        : deesse.CONNECT_FACE_EDGE,
        'connect_face_edge_corner' : deesse.CONNECT_FACE_EDGE_CORNER
    }
    try:
        co_c.connectivityType = connectivityType_dict[co_py.connectivityType]
    except:
        # Free memory on C side
        deesse.MPDSFreeConnectivity(co_c)
        deesse.free_MPDS_CONNECTIVITY(co_c)
        err_msg = f'{fname}: connectivity type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # ... varName
    deesse.mpds_allocate_and_set_connectivity_varname(co_c, co_py.varname)

    # ... classOfValues
    co_c.classOfValues = classInterval2classOfValues(co_py.classInterval)

    # ... tiAsRefFlag
    if co_py.tiAsRefFlag:
        co_c.tiAsRefFlag = deesse.TRUE
    else:
        co_c.tiAsRefFlag = deesse.FALSE

    if not co_py.tiAsRefFlag:
        # ... refConnectivityImage
        im = img.copyImg(co_py.refConnectivityImage, logger=logger)
        im.extract_var([co_py.refConnectivityVarIndex], logger=logger)
        try:
            co_c.refConnectivityImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeConnectivity(co_c)
            deesse.free_MPDS_CONNECTIVITY(co_c)
            err_msg = f"{fname}: cannot convert connectivity parameters from python to C ('refConnectivityImage')"
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

    # ... deactivationDistance
    co_c.deactivationDistance = co_py.deactivationDistance

    # ... threshold
    co_c.threshold = co_py.threshold

    return co_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def connectivity_C2py(co_c, logger=None):
    """
    Converts connectivity parameters from C to python.

    Parameters
    ----------
    co_c : \\(MPDS_CONNECTIVITY \\*\\)
        connectivity parameters in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    co_py : :class:`Connectivity`
        connectivity parameters in python
    """
    fname = 'connectivity_C2py'

    # ... connectivityConstraintUsage
    connectivityConstraintUsage = co_c.connectivityConstraintUsage
    if connectivityConstraintUsage == 0:
        co_py = Connectivity()
        return co_py

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
        err_msg = f'{fname}: connectivity type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # ... varName
    varname = co_c.varName

    # ... classInterval
    classInterval = classOfValues2classInterval(co_c.classOfValues)
    nclass = len(classInterval)

    # ... tiAsRefFlag
    tiAsRefFlag = bool(int.from_bytes(co_c.tiAsRefFlag.encode('utf-8'), byteorder='big'))

    if not tiAsRefFlag:
        # ... refConnectivityImage
        refConnectivityImage = img_C2py(co_c.refConnectivityImage, logger=logger)
        refConnectivityVarIndex = 0

    # ... deactivationDistance
    deactivationDistance = co_c.deactivationDistance

    # ... threshold
    threshold = co_c.threshold

    co_py = Connectivity(
        connectivityConstraintUsage=connectivityConstraintUsage,
        connectivityType=connectivityType,
        nclass=nclass,
        classInterval=classInterval,
        varname=varname,
        tiAsRefFlag=tiAsRefFlag,
        refConnectivityImage=refConnectivityImage,
        refConnectivityVarIndex=refConnectivityVarIndex,
        deactivationDistance=deactivationDistance,
        threshold=threshold,
        logger=logger
    )

    return co_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def blockData_py2C(bd_py):
    """
    Converts block data parameters from python to C.

    Parameters
    ----------
    bd_py : :class:`BlockData`
        block data parameters in python

    Returns
    -------
    bd_c : \\(MPDS_BLOCKDATA \\*\\)
        block data parameters in C
    """
    # fname = 'blockData_py2C'

    bd_c = deesse.malloc_MPDS_BLOCKDATA()
    deesse.MPDSInitBlockData(bd_c)

    # ... blockDataUsage
    bd_c.blockDataUsage = bd_py.blockDataUsage
    if bd_py.blockDataUsage == 0:
        return bd_c

    # ... nblock
    bd_c.nblock = bd_py.nblock

    # ... nnode, ix, iy, iz
    bd_c.nnode = deesse.new_int_array(bd_py.nblock)
    bd_c.ix = deesse.new_intp_array(bd_py.nblock)
    bd_c.iy = deesse.new_intp_array(bd_py.nblock)
    bd_c.iz = deesse.new_intp_array(bd_py.nblock)

    for j, ni in enumerate(bd_py.nodeIndex):
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
    bd_c.value = deesse.new_real_array(bd_py.nblock)
    deesse.mpds_set_real_vector_from_array(bd_c.value, 0,
        np.asarray(bd_py.value).reshape(bd_py.nblock))

    # ... tolerance
    bd_c.tolerance = deesse.new_real_array(bd_py.nblock)
    deesse.mpds_set_real_vector_from_array(bd_c.tolerance, 0,
        np.asarray(bd_py.tolerance).reshape(bd_py.nblock))

    # ... activatePropMin
    bd_c.activatePropMin = deesse.new_real_array(bd_py.nblock)
    deesse.mpds_set_real_vector_from_array(bd_c.activatePropMin, 0,
        np.asarray(bd_py.activatePropMin).reshape(bd_py.nblock))

    # ... activatePropMax
    bd_c.activatePropMax = deesse.new_real_array(bd_py.nblock)
    deesse.mpds_set_real_vector_from_array(bd_c.activatePropMax, 0,
        np.asarray(bd_py.activatePropMax).reshape(bd_py.nblock))

    return bd_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def blockData_C2py(bd_c):
    """
    Converts block data parameters from C to python.

    Parameters
    ----------
    bd_c : \\(MPDS_BLOCKDATA \\*\\)
        block data parameters in C

    Returns
    -------
    bd_py : :class:`BlockData`
        block data parameters in python
    """
    # fname = 'blockData_C2py'

    # ... blockDataUsage
    blockDataUsage = bd_c.blockDataUsage
    if blockDataUsage == 0:
        bd_py = BlockData()
        return bd_py

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

    bd_py = BlockData(
         blockDataUsage=blockDataUsage,
         nblock=nblock,
         nodeIndex=nodeIndex,
         value=value,
         tolerance=tolerance,
         activatePropMin=activatePropMin,
         activatePropMax=activatePropMax
    )

    return bd_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidGeneralParameters_py2C(pgp_py):
    """
    Converts pyramid general parameters from python to C.

    Parameters
    ----------
    pgp_py : :class:`PyramidGeneralParameters`
        pyramid general parameters in python

    Returns
    -------
    pgp_c : \\(MPDS_PYRAMIDGENERALPARAMETERS \\*\\)
        pyramid general parameters in C
    """
    # fname = 'pyramidGeneralParameters_py2C'

    pgp_c = deesse.malloc_MPDS_PYRAMIDGENERALPARAMETERS()
    deesse.MPDSInitPyramidGeneralParameters(pgp_c)

    # ... npyramidLevel
    nl = int(pgp_py.npyramidLevel)
    pgp_c.npyramidLevel = nl

    # ... pyramidSimulationMode
    pyramidSimulationMode_dict = {
        'hierarchical'                 : deesse.PYRAMID_SIM_HIERARCHICAL,
        'hierarchical_using_expansion' : deesse.PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION,
        'all_level_one_by_one'         : deesse.PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE,
        'pyramid_sim_none'             : deesse.PYRAMID_SIM_NONE
    }
    try:
        pgp_c.pyramidSimulationMode = pyramidSimulationMode_dict[pgp_py.pyramidSimulationMode]
    except:
        pgp_c.pyramidSimulationMode = pyramidSimulationMode_dict['pyramid_sim_none']

    if nl > 0:
        # ... kx
        pgp_c.kx = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
                pgp_c.kx, 0,
                np.asarray(pgp_py.kx, dtype='intc').reshape(nl))

        # ... ky
        pgp_c.ky = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
                pgp_c.ky, 0,
                np.asarray(pgp_py.ky, dtype='intc').reshape(nl))

        # ... kz
        pgp_c.kz = deesse.new_int_array(nl)
        deesse.mpds_set_int_vector_from_array(
                pgp_c.kz, 0,
                np.asarray(pgp_py.kz, dtype='intc').reshape(nl))

        # ... factorNneighboringNode and factorDistanceThreshold ...
        if pgp_py.pyramidSimulationMode in ('hierarchical', 'hierarchical_using_expansion'):
            nn = 4*nl + 1
        else: # pyramidSimulationMode == 'all_level_one_by_one'
            nn = nl + 1

        # ... factorNneighboringNode
        pgp_c.factorNneighboringNode = deesse.new_double_array(nn)
        deesse.mpds_set_double_vector_from_array(
                pgp_c.factorNneighboringNode, 0,
                np.asarray(pgp_py.factorNneighboringNode).reshape(nn))

        # ... factorDistanceThreshold
        pgp_c.factorDistanceThreshold = deesse.new_real_array(nn)
        deesse.mpds_set_real_vector_from_array(
                pgp_c.factorDistanceThreshold, 0,
                np.asarray(pgp_py.factorDistanceThreshold).reshape(nn))

        # ... factorMaxScanFraction
        pgp_c.factorMaxScanFraction = deesse.new_double_array(nl+1)
        deesse.mpds_set_double_vector_from_array(
                pgp_c.factorMaxScanFraction, 0,
                np.asarray(pgp_py.factorMaxScanFraction).reshape(nl+1))

    return pgp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidGeneralParameters_C2py(pgp_c, logger=None):
    """
    Converts pyramid general parameters from C to python.

    Parameters
    ----------
    pgp_c : \\(MPDS_PYRAMIDGENERALPARAMETERS \\*\\)
        pyramid general parameters in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    pgp_py : :class:`PyramidGeneralParameters`
        pyramid general parameters in python
    """
    # fname = 'pyramidGeneralParameters_C2py'

    # ... npyramidLevel
    npyramidLevel = pgp_c.npyramidLevel
    if npyramidLevel == 0:
        pgp_py = PyramidGeneralParameters()
        return pgp_py

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

    pgp_py = PyramidGeneralParameters(
        npyramidLevel=npyramidLevel,
        kx=kx, ky=ky, kz=kz,
        pyramidSimulationMode=pyramidSimulationMode,
        factorNneighboringNode=factorNneighboringNode,
        factorDistanceThreshold=factorDistanceThreshold,
        factorMaxScanFraction=factorMaxScanFraction,
        logger=logger
    )

    return pgp_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidParameters_py2C(pp_py, logger=None):
    """
    Converts pyramid parameters from python to C.

    Parameters
    ----------
    pp_py : :class:`PyramidParameters`
        pyramid parameters in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    pp_c : \\(MPDS_PYRAMIDPARAMETERS \\*\\)
        pyramid parameters in C
    """
    fname = 'pyramidParameters_py2C'

    pp_c = deesse.malloc_MPDS_PYRAMIDPARAMETERS()
    deesse.MPDSInitPyramidParameters(pp_c)

    # ... nlevel
    pp_c.nlevel = int(pp_py.nlevel)

    # ... pyramidType
    pyramidType_dict = {
        'none'                      : deesse.PYRAMID_NONE,
        'continuous'                : deesse.PYRAMID_CONTINUOUS,
        'categorical_auto'          : deesse.PYRAMID_CATEGORICAL_AUTO,
        'categorical_custom'        : deesse.PYRAMID_CATEGORICAL_CUSTOM,
        'categorical_to_continuous' : deesse.PYRAMID_CATEGORICAL_TO_CONTINUOUS
    }
    try:
        pp_c.pyramidType = pyramidType_dict[pp_py.pyramidType]
    except:
        # Free memory on C side
        deesse.MPDSFreePyramidParameters(pp_c)
        deesse.free_MPDS_PYRAMIDPARAMETERS(pp_c)
        err_msg = f'{fname}: pyramid type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    if pp_py.pyramidType == 'categorical_custom':
        # ... classOfValues
        pp_c.classOfValues = classInterval2classOfValues(pp_py.classInterval)

    # ... outputLevelFlag
    deesse.mpds_allocate_and_set_pyramid_outputLevelFlag(pp_c, np.array([int(i) for i in pp_py.outputLevelFlag], dtype='intc'))

    return pp_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pyramidParameters_C2py(pp_c, logger=None):
    """
    Converts pyramid parameters from C to python.

    Parameters
    ----------
    pp_c : \\(MPDS_PYRAMIDPARAMETERS \\*\\)
        pyramid parameters in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    pp_py : :class:`PyramidParameters`
        pyramid parameters in python
    """
    fname = 'pyramidParameters_C2py'

    # ... nlevel
    nlevel = pp_c.nlevel
    if nlevel == 0:
        pp_py = PyramidParameters()
        return pp_py

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
        err_msg = f'{fname}: pyramid type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    if pyramidType == 'categorical_custom':
        # ... classInterval
        classInterval = classOfValues2classInterval(pp_c.classOfValues)
        nclass = len(classInterval)

    # ... outputLevelFlag
    outputLevelFlag = np.zeros(nlevel, dtype='intc')
    deesse.mpds_get_pyramid_outputLevelFlag(pp_c, outputLevelFlag)
    outputLevelFlag = outputLevelFlag.astype('bool')

    pp_py = PyramidParameters(
         nlevel=nlevel,
         pyramidType=pyramidType,
         nclass=nclass,
         classInterval=classInterval,
         outputLevelFlag=outputLevelFlag,
         logger=logger
    )

    return pp_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_input_py2C(deesse_input, logger=None):
    """
    Converts deesse input from python to C.

    Parameters
    ----------
    deesse_input : :class:`DeesseInput`
        deesse input in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_siminput : \\(MPDS_SIMINPUT \\*\\)
        deesse input in C
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
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: simName is not a string'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    if len(deesse_input.simName) >= deesse.MPDS_VARNAME_LENGTH:
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: simName is too long'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    deesse.mpds_allocate_and_set_simname(mpds_siminput, deesse_input.simName)
    # mpds_siminput.simName = deesse_input.simName #  works too

    # mpds_siminput.simImage ...
    # ... set initial image im (for simulation)
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=nv, val=deesse.MPDS_MISSING_VALUE,
             varname=deesse_input.varname,
             logger=logger)

    # ... convert im from python to C
    try:
        mpds_siminput.simImage = img_py2C(im, logger=logger)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: cannot initialize simImage in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # if mpds_siminput.simImage is None:
    #     # Free memory on C side
    #     deesse.MPDSFreeSimInput(mpds_siminput)
    #     deesse.free_MPDS_SIMINPUT(mpds_siminput)
    #     return None

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
            try:
                im_c = img_py2C(ti, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert TI from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.trainImage, i, im_c)

    # mpds_siminput.pdfTrainImage
    if nTI > 1:
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=nTI, val=deesse_input.pdfTI,
                 logger=logger)
        try:
            mpds_siminput.pdfTrainImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot convert pdfTI from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

    # mpds_siminput.ndataImage and mpds_siminput.dataImage
    if deesse_input.dataImage is None:
        mpds_siminput.ndataImage = 0
    else:
        n = len(deesse_input.dataImage)
        mpds_siminput.ndataImage = n
        mpds_siminput.dataImage = deesse.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(deesse_input.dataImage):
            try:
                im_c = img_py2C(dataIm, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert dataImage from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImage, i, im_c)

    # mpds_siminput.ndataPointSet and mpds_siminput.dataPointSet
    if deesse_input.dataPointSet is None:
        mpds_siminput.ndataPointSet = 0
    else:
        n = len(deesse_input.dataPointSet)
        mpds_siminput.ndataPointSet = n
        mpds_siminput.dataPointSet = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesse_input.dataPointSet):
            try:
                ps_c = ps_py2C(dataPS, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert dataPointSet from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSet, i, ps_c)

    # mpds_siminput.maskImageFlag and mpds_siminput.maskImage
    if deesse_input.mask is None:
        mpds_siminput.maskImageFlag = deesse.FALSE
    else:
        mpds_siminput.maskImageFlag = deesse.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=deesse_input.mask,
                 logger=logger)
        try:
            mpds_siminput.maskImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot convert mask from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.homothetyXRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyXRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyXRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.homothetyYRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyYRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyYRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.homothetyZRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyZRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyZRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.homothetyXRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyXRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyXRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.homothetyYRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyYRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyYRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.homothetyZRatio,
                     logger=logger)
            try:
                mpds_siminput.homothetyZRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert homothetyZRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.rotationAzimuth,
                     logger=logger)
            try:
                mpds_siminput.rotationAzimuthImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationAzimuth image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.rotationDip,
                     logger=logger)
            try:
                mpds_siminput.rotationDipImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationDip image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=deesse_input.rotationPlunge,
                     logger=logger)
            try:
                mpds_siminput.rotationPlungeImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationPlunge image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.rotationAzimuth,
                     logger=logger)
            try:
                mpds_siminput.rotationAzimuthImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationAzimuth image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.rotationDip,
                     logger=logger)
            try:
                mpds_siminput.rotationDipImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationDip image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=deesse_input.rotationPlunge,
                     logger=logger)
            try:
                mpds_siminput.rotationPlungeImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert rotationPlunge image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: normalizing type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # mpds_siminput.searchNeighborhoodParameters
    mpds_siminput.searchNeighborhoodParameters = deesse.new_MPDS_SEARCHNEIGHBORHOODPARAMETERS_array(nv)
    for i, sn in enumerate(deesse_input.searchNeighborhoodParameters):
        try:
            sn_c = search_neighborhood_parameters_py2C(sn, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot convert search neighborhood parameters from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: rescaling mode unknown'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
    try:
        mpds_siminput.simAndPathParameters = set_simAndPathParameters_C(
                deesse_input.simType,
                deesse_input.simPathType,
                deesse_input.simPathStrength,
                deesse_input.simPathPower,
                deesse_input.simPathUnilateralOrder,
                logger=logger)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: cannot set "simAndPathParameters" in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # mpds_siminput.distanceThreshold
    mpds_siminput.distanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_siminput.distanceThreshold, 0,
        np.asarray(deesse_input.distanceThreshold).reshape(nv))

    # mpds_siminput.softProbability ...
    mpds_siminput.softProbability = deesse.new_MPDS_SOFTPROBABILITY_array(nv)

    # ... for each variable ...
    for i, sp in enumerate(deesse_input.softProbability):
        try:
            sp_c = softProbability_py2C(sp,
                                        nx, ny, nz,
                                        sx, sy, sz,
                                        ox, oy, oz,
                                        logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot set soft probability parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

        deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_siminput.softProbability, i, sp_c)
        # deesse.free_MPDS_SOFTPROBABILITY(sp_c)

    # mpds_siminput.connectivity ...
    mpds_siminput.connectivity = deesse.new_MPDS_CONNECTIVITY_array(nv)

    for i, co in enumerate(deesse_input.connectivity):
        try:
            co_c = connectivity_py2C(co, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot set connectivity parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

        deesse.MPDS_CONNECTIVITY_array_setitem(mpds_siminput.connectivity, i, co_c)
        # deesse.free_MPDS_CONNECTIVITY(co_c)

    # mpds_siminput.blockData ...
    mpds_siminput.blockData = deesse.new_MPDS_BLOCKDATA_array(nv)
    # ... for each variable ...
    for i, bd in enumerate(deesse_input.blockData):
        try:
            bd_c = blockData_py2C(bd)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot set block data parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

        deesse.MPDS_BLOCKDATA_array_setitem(mpds_siminput.blockData, i, bd_c)
        # deesse.free_MPDS_BLOCKDATA(bd_c)

    # mpds_siminput.maxScanFraction
    mpds_siminput.maxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_siminput.maxScanFraction, 0,
            np.asarray(deesse_input.maxScanFraction).reshape(nTI))

    # mpds_siminput.pyramidGeneralParameters ...
    try:
        mpds_siminput.pyramidGeneralParameters = pyramidGeneralParameters_py2C(deesse_input.pyramidGeneralParameters)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        err_msg = f'{fname}: cannot set pyramid general parameters in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # mpds_siminput.pyramidParameters ...
    mpds_siminput.pyramidParameters = deesse.new_MPDS_PYRAMIDPARAMETERS_array(nv)

    # ... for each variable ...
    for i, pp in enumerate(deesse_input.pyramidParameters):
        try:
            pp_c = pyramidParameters_py2C(pp, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeSimInput(mpds_siminput)
            deesse.free_MPDS_SIMINPUT(mpds_siminput)
            err_msg = f'{fname}: cannot set pyramid parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
            try:
                im_c = img_py2C(dataIm, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert pyramidDataImage from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_IMAGE_array_setitem(mpds_siminput.dataImageInPyramid, i, im_c)

    # mpds_siminput.ndataPointSetInPyramid and mpds_siminput.dataPointSetInPyramid
    if deesse_input.pyramidDataPointSet is None:
        mpds_siminput.ndataPointSetInPyramid = 0
    else:
        n = len(deesse_input.pyramidDataPointSet)
        mpds_siminput.ndataPointSetInPyramid = n
        mpds_siminput.dataPointSetInPyramid = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesse_input.pyramidDataPointSet):
            try:
                ps_c = ps_py2C(dataPS, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeSimInput(mpds_siminput)
                deesse.free_MPDS_SIMINPUT(mpds_siminput)
                err_msg = f'{fname}: cannot convert pyramidDataPointSet from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_POINTSET_array_setitem(mpds_siminput.dataPointSetInPyramid, i, ps_c)

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
def deesse_input_C2py(mpds_siminput, logger=None):
    """
    Converts deesse input from C to python.

    Parameters
    ----------
    mpds_siminput : \\(MPDS_SIMINPUT \\*\\)
        deesse input in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesse_input :class:`DeesseInput`
        deesse input in python
    """
    fname = 'deesse_input_C2py'

    # simName
    simName = mpds_siminput.simName

    im = img_C2py(mpds_siminput.simImage, logger=logger)

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
            TI[i] = img_C2py(im, logger=logger)

    # pdfTI
    pdfTI = None
    if nTI > 1:
        im = img_C2py(mpds_siminput.pdfTrainImage, logger=logger)
        pdfTI = im.val

    # dataImage
    dataImage = None
    ndataImage = mpds_siminput.ndataImage
    if ndataImage > 0:
        dataImage = np.array(ndataImage*[None])
        for i in range(ndataImage):
            im = deesse.MPDS_IMAGE_array_getitem(mpds_siminput.dataImage, i)
            dataImage[i] = img_C2py(im, logger=logger)

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
        im = img_C2py(mpds_siminput.maskImage, logger=logger)
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
            im = img_C2py(mpds_siminput.homothetyXRatioImage, logger=logger)
            homothetyXRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyXRatioValue, 0, v)
            homothetyXRatio = v[0]

        homothetyYLocal = bool(int.from_bytes(mpds_siminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_siminput.homothetyYRatioImage, logger=logger)
            homothetyYRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyYRatioValue, 0, v)
            homothetyYRatio = v[0]

        homothetyZLocal = bool(int.from_bytes(mpds_siminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_siminput.homothetyZRatioImage, logger=logger)
            homothetyZRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyZRatioValue, 0, v)
            homothetyZRatio = v[0]

    elif homothetyUsage == 2:
        homothetyXLocal = bool(int.from_bytes(mpds_siminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_siminput.homothetyXRatioImage, logger=logger)
            homothetyXRatio = im.val
        else:
            homothetyXRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyXRatioValue, 0, homothetyXRatio)

        homothetyYLocal = bool(int.from_bytes(mpds_siminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_siminput.homothetyYRatioImage, logger=logger)
            homothetyYRatio = im.val
        else:
            homothetyYRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.homothetyYRatioValue, 0, homothetyYRatio)

        homothetyZLocal = bool(int.from_bytes(mpds_siminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_siminput.homothetyZRatioImage, logger=logger)
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
            im = img_C2py(mpds_siminput.rotationAzimuthImage, logger=logger)
            rotationAzimuth = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationAzimuthValue, 0, v)
            rotationAzimuth = v[0]

        rotationDipLocal = bool(int.from_bytes(mpds_siminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_siminput.rotationDipImage, logger=logger)
            rotationDip = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationDipValue, 0, v)
            rotationDip = v[0]

        rotationPlungeLocal = bool(int.from_bytes(mpds_siminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_siminput.rotationPlungeImage, logger=logger)
            rotationPlunge = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationPlungeValue, 0, v)
            rotationPlunge = v[0]

    elif rotationUsage == 2:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_siminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_siminput.rotationAzimuthImage, logger=logger)
            rotationAzimuth = im.val
        else:
            rotationAzimuth = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationAzimuthValue, 0, rotationAzimuth)

        rotationDipLocal = bool(int.from_bytes(mpds_siminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_siminput.rotationDipImage, logger=logger)
            rotationDip = im.val
        else:
            rotationDip = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_siminput.rotationDipValue, 0, rotationDip)

        rotationPlungeLocal = bool(int.from_bytes(mpds_siminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_siminput.rotationPlungeImage, logger=logger)
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
        err_msg = f'{fname}: normalizing type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # searchNeighborhoodParameters
    searchNeighborhoodParameters = np.array(nv*[None])
    for i in range(nv):
        sn_c = deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_getitem(mpds_siminput.searchNeighborhoodParameters, i)
        sn = search_neighborhood_parameters_C2py(sn_c, logger=logger)
        if sn is None:
            err_msg = f'{fname}: cannot convert search neighborhood parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
            err_msg = f'{fname}: rescaling mode unknown'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
        err_msg = f'{fname}: simulation type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        err_msg = f'{fname}: simulation path type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # distanceThreshold
    distanceThreshold = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_siminput.distanceThreshold, 0, distanceThreshold)

    # softProbability
    softProbability = np.array(nv*[None])
    for i in range(nv):
        sp_c = deesse.MPDS_SOFTPROBABILITY_array_getitem(mpds_siminput.softProbability, i)
        sp = softProbability_C2py(sp_c, logger=logger)
        if sp is None:
            err_msg = f'{fname}: cannot convert soft probability from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        softProbability[i] = sp

    # connectivity
    connectivity = np.array(nv*[None])
    for i in range(nv):
        co_c = deesse.MPDS_CONNECTIVITY_array_getitem(mpds_siminput.connectivity, i)
        co = connectivity_C2py(co_c, logger=logger)
        if co is None:
            err_msg = f'{fname}: cannot convert connectivity parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        connectivity[i] = co

    # blockData
    blockData = np.array(nv*[None])
    for i in range(nv):
        bd_c = deesse.MPDS_BLOCKDATA_array_getitem(mpds_siminput.blockData, i)
        bd = blockData_C2py(bd_c)
        if bd is None:
            err_msg = f'{fname}: cannot convert block data parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        blockData[i] = bd

    # maxScanFraction
    maxScanFraction = np.zeros(nTI, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_siminput.maxScanFraction, 0, maxScanFraction)
    maxScanFraction = maxScanFraction.astype('float')

    # pyramidGeneralParameters
    pyramidGeneralParameters = pyramidGeneralParameters_C2py(mpds_siminput.pyramidGeneralParameters, logger=logger)
    if pyramidGeneralParameters is None:
        err_msg = f'{fname}: cannot convert pyramid general parameters from C to python'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # pyramidParameters
    pyramidParameters = np.array(nv*[None])
    for i in range(nv):
        pp_c = deesse.MPDS_PYRAMIDPARAMETERS_array_getitem(mpds_siminput.pyramidParameters, i)
        pp = pyramidParameters_C2py(pp_c, logger=logger)
        if pp is None:
            err_msg = f'{fname}: cannot convert pyramid parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        pyramidParameters[i] = pp

    # pyramidDataImage
    pyramidDataImage = None
    npyramidDataImage = mpds_siminput.ndataImageInPyramid
    if npyramidDataImage > 0:
        pyramidDataImage = np.array(npyramidDataImage*[None])
        for i in range(npyramidDataImage):
            im = deesse.MPDS_IMAGE_array_getitem(mpds_siminput.dataImageInPyramid, i)
            pyramidDataImage[i] = img_C2py(im, logger=logger)

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
        nrealization=nrealization,
        logger=logger)

    return deesse_input
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesse_output_C2py(mpds_simoutput, mpds_progressMonitor, logger=None):
    """
    Converts deesse output from C to python.

    Parameters
    ----------
    mpds_simoutput : \\(MPDS_SIMOUTPUT \\*\\)
        deesse output in C

    mpds_progressMonitor : \\(MPDS_PROGRESSMONITOR \\*\\)
        progress monitor in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesse_output : dict
        deesse output in python, dictionary

        `{'sim':sim,
        'sim_var_original_index':sim_var_original_index,
        'sim_pyramid':sim_pyramid,
        'sim_pyramid_var_original_index':sim_pyramid_var_original_index,
        'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
        'path':path,
        'error':error,
        'tiGridNode':tiGridNode,
        'tiIndex':tiIndex,
        'nwarning':nwarning,
        'warnings':warnings}`

        with (`nreal=mpds_simoutput->nreal`, the number of realization(s)):

        - sim: 1D array of :class:`geone.img.Img` of shape (nreal,)
            * `sim[i]`: i-th realisation, \
            k-th variable stored refers to the original variable \
            `sim_var_original_index[k]` \
            (get from `mpds_simoutput->outputSimImage[0]`)

            note: `sim=None` if `mpds_simoutput->outputSimImage=NULL`

        - sim_var_original_index : 1D array of ints
            * `sim_var_original_index[k]`: index of the original variable \
            (given in deesse_input) of the k-th variable stored in \
            in `sim[i]` for any i (array of length `sim[*].nv`, \
            get from `mpds_simoutput->originalVarIndex`)

            note: `sim_var_original_index=None`
            if `mpds_simoutput->originalVarIndex=NULL`

        - sim_pyramid : list, optional
            realizations in pyramid levels (depends on input parameters given in
            deesse_input); if pyramid was used and output in pyramid required:

            * `sim_pyramid[j]` : 1D array of `nreal` `:class:`geone.img.Img` \
            (or None):

                - `sim_pyramid[j][i]`: i-th realisation in pyramid level of \
                index j+1, k-th variable stored refers to \
                the original variable `sim_pyramid_var_original_index[j][k]` \
                and pyramid index `sim_pyramid_var_pyramid_index[j][k]` \
                (get from `mpds_simoutput->outputSimImagePyramidLevel[j]`)

            note: `sim_pyramid[j]=None` if
            `mpds_simoutput->outputSimImagePyramidLevel[j]=NULL`

        - sim_pyramid_var_original_index : list, optional
            index of original variable for realizations in pyramid levels \
            (depends on input parameters given in deesse_input); if pyramid was \
            used and output in pyramid required:

            * `sim_pyramid_var_original_index[j]` : 1D array of ints (or None):

                - `sim_pyramid_var_original_index[j][k]`: index of the \
                original variable (given in deesse_input) of the k-th variable \
                stored in `sim_pyramid[j][i]`, for any i \
                (array of length sim_pyramid[j][*].nv, get from \
                `mpds_simoutput->originalVarIndexPyramidLevel[j]`)

            note: `sim_pyramid_var_original_index[j]=None` if
            `mpds_simoutput->originalVarIndexPyramidLevel[j]=NULL`

        - sim_pyramid_var_pyramid_index : list, optional
            pyramid index of original variable for realizations in pyramid levels
            (depends on input parameters given in deesse_input); if pyramid was
            used and output in pyramid required:

            * `sim_pyramid_var_pyramid_index[j]` : 1D array of ints (or None):

                - `sim_pyramid_var_pyramid_index[j][k]`: pyramid index of \
                original variable (given in deesse_input) of the k-th variable \
                stored in `sim_pyramid[j][i]`, for any i \
                (array of length `sim_pyramid[j][*].nv`, get from \
                `mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]`)

            note: `sim_pyramid_var_pyramid_index[j]=None` if
            `mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]=NULL`

        - path : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `path[i]`: path index map for the i-th realisation \
            (`mpds_simoutput->outputPathIndexImage[0]`)

            note: `path=None` if `mpds_simoutput->outputPathIndexImage=NULL`

        - error : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            `error[i]`: error map for the i-th realisation
            (`mpds_simoutput->outputErrorImage[0]`)
            note: `error=None` if `mpds_simoutput->outputErrorImage=NULL`

        - tiGridNode : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `tiGridNode[i]`: TI grid node index map for the i-th realisation \
            (`mpds_simoutput->outputTiGridNodeIndexImage[0]`)

            note: `tiGridNode=None` if `mpds_simoutput->outputTiGridNodeIndexImage=NULL`

        - tiIndex : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `tiIndex[i]`: TI index map for the i-th realisation \
            (`mpds_simoutput->outputTiIndexImage[0]`)

            note: `tiIndex=None` if `mpds_simoutput->outputTiIndexImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    # fname = 'deesse_output_C2py'

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
            im = img_C2py(mpds_simoutput.outputSimImage, logger=logger)

            nv = mpds_simoutput.nvarSimPerReal
            k = 0
            sim = []
            for i in range(nreal):
                sim.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                               sx=im.sx, sy=im.sy, sz=im.sz,
                               ox=im.ox, oy=im.oy, oz=im.oz,
                               nv=nv, val=im.val[k:(k+nv),...],
                               varname=im.varname[k:(k+nv)],
                               logger=logger))
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
                            im = img_C2py(im_ptr, logger=logger)

                            nv = nvarSimPerRealPyramidLevel[j]
                            k = 0
                            sim_pyramid[j] = []
                            for i in range(nreal):
                                sim_pyramid[j].append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                               sx=im.sx, sy=im.sy, sz=im.sz,
                                               ox=im.ox, oy=im.oy, oz=im.oz,
                                               nv=nv, val=im.val[k:(k+nv),...],
                                               varname=im.varname[k:(k+nv)],
                                               logger=logger))
                                k = k + nv

                            del(im)
                            sim_pyramid[j] = np.asarray(sim_pyramid[j]).reshape(nreal)
                            # +++
                    # ---

        if mpds_simoutput.nvarPathIndexPerReal:
            # --- path ---
            im = img_C2py(mpds_simoutput.outputPathIndexImage, logger=logger)

            nv = mpds_simoutput.nvarPathIndexPerReal
            k = 0
            path = []
            for i in range(nreal):
                path.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)],
                                logger=logger))
                k = k + nv

            del(im)
            path = np.asarray(path).reshape(nreal)
            # ---

        if mpds_simoutput.nvarErrorPerReal:
            # --- error ---
            im = img_C2py(mpds_simoutput.outputErrorImage, logger=logger)

            nv = mpds_simoutput.nvarErrorPerReal
            k = 0
            error = []
            for i in range(nreal):
                error.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                 sx=im.sx, sy=im.sy, sz=im.sz,
                                 ox=im.ox, oy=im.oy, oz=im.oz,
                                 nv=nv, val=im.val[k:(k+nv),...],
                                 varname=im.varname[k:(k+nv)],
                                 logger=logger))
                k = k + nv

            del(im)
            error = np.asarray(error).reshape(nreal)
            # ---

        if mpds_simoutput.nvarTiGridNodeIndexPerReal:
            # --- tiGridNode ---
            im = img_C2py(mpds_simoutput.outputTiGridNodeIndexImage, logger=logger)

            nv = mpds_simoutput.nvarTiGridNodeIndexPerReal
            k = 0
            tiGridNode = []
            for i in range(nreal):
                tiGridNode.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)],
                                logger=logger))
                k = k + nv

            del(im)
            tiGridNode = np.asarray(tiGridNode).reshape(nreal)
            # ---

        if mpds_simoutput.nvarTiIndexPerReal:
            # --- tiIndex ---
            im = img_C2py(mpds_simoutput.outputTiIndexImage, logger=logger)

            nv = mpds_simoutput.nvarTiIndexPerReal
            k = 0
            tiIndex = []
            for i in range(nreal):
                tiIndex.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                sx=im.sx, sy=im.sy, sz=im.sz,
                                ox=im.ox, oy=im.oy, oz=im.oz,
                                nv=nv, val=im.val[k:(k+nv),...],
                                varname=im.varname[k:(k+nv)],
                                logger=logger))
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
def deesseRun(
        deesse_input,
        add_data_point_to_mask=True,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Launches deesse.

    Parameters
    ----------
    deesse_input : :class:`DeesseInput`
        deesse input in python

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored<

    nthreads : int, default: -1
        number of thread(s) to use for C program;
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
    deesse_output : dict
        deesse output in python, dictionary

        `{'sim':sim,
        'sim_var_original_index':sim_var_original_index,
        'sim_pyramid':sim_pyramid,
        'sim_pyramid_var_original_index':sim_pyramid_var_original_index,
        'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
        'path':path,
        'error':error,
        'tiGridNode':tiGridNode,
        'tiIndex':tiIndex,
        'nwarning':nwarning,
        'warnings':warnings}`

        with `nreal=deesse_input.nrealization`:

        - sim: 1D array of :class:`geone.img.Img` of shape (nreal,)
            * `sim[i]`: i-th realisation, \
            k-th variable stored refers to the original variable \
            `sim_var_original_index[k]` \
            (get from `mpds_simoutput->outputSimImage[0]`)

            note: `sim=None` if `mpds_simoutput->outputSimImage=NULL`

        - sim_var_original_index : 1D array of ints
            * `sim_var_original_index[k]`: index of the original variable \
            (given in deesse_input) of the k-th variable stored in \
            in `sim[i]` for any i (array of length `sim[*].nv`, \
            get from `mpds_simoutput->originalVarIndex`)

            note: `sim_var_original_index=None`
            if `mpds_simoutput->originalVarIndex=NULL`

        - sim_pyramid : list, optional
            realizations in pyramid levels (depends on input parameters given in
            deesse_input); if pyramid was used and output in pyramid required:

            * `sim_pyramid[j]` : 1D array of `nreal` `:class:`geone.img.Img` \
            (or None):

                - `sim_pyramid[j][i]`: i-th realisation in pyramid level of \
                index j+1, k-th variable stored refers to \
                the original variable `sim_pyramid_var_original_index[j][k]` \
                and pyramid index `sim_pyramid_var_pyramid_index[j][k]` \
                (get from `mpds_simoutput->outputSimImagePyramidLevel[j]`)

            note: `sim_pyramid[j]=None` if
            `mpds_simoutput->outputSimImagePyramidLevel[j]=NULL`

        - sim_pyramid_var_original_index : list, optional
            index of original variable for realizations in pyramid levels \
            (depends on input parameters given in deesse_input); if pyramid was \
            used and output in pyramid required:

            * `sim_pyramid_var_original_index[j]` : 1D array of ints (or None):

                - `sim_pyramid_var_original_index[j][k]`: index of the \
                original variable (given in deesse_input) of the k-th variable \
                stored in `sim_pyramid[j][i]`, for any i \
                (array of length sim_pyramid[j][*].nv, get from \
                `mpds_simoutput->originalVarIndexPyramidLevel[j]`)

            note: `sim_pyramid_var_original_index[j]=None` if
            `mpds_simoutput->originalVarIndexPyramidLevel[j]=NULL`

        - sim_pyramid_var_pyramid_index : list, optional
            pyramid index of original variable for realizations in pyramid levels
            (depends on input parameters given in deesse_input); if pyramid was
            used and output in pyramid required:

            * `sim_pyramid_var_pyramid_index[j]` : 1D array of ints (or None):

                - `sim_pyramid_var_pyramid_index[j][k]`: pyramid index of \
                original variable (given in deesse_input) of the k-th variable \
                stored in `sim_pyramid[j][i]`, for any i \
                (array of length `sim_pyramid[j][*].nv`, get from \
                `mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]`)

            note: `sim_pyramid_var_pyramid_index[j]=None` if
            `mpds_simoutput->pyramidIndexOfOriginalVarPyramidLevel[j]=NULL`

        - path : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `path[i]`: path index map for the i-th realisation \
            (`mpds_simoutput->outputPathIndexImage[0]`)

            note: `path=None` if `mpds_simoutput->outputPathIndexImage=NULL`

        - error : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            `error[i]`: error map for the i-th realisation
            (`mpds_simoutput->outputErrorImage[0]`)
            note: `error=None` if `mpds_simoutput->outputErrorImage=NULL`

        - tiGridNode : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `tiGridNode[i]`: TI grid node index map for the i-th realisation \
            (`mpds_simoutput->outputTiGridNodeIndexImage[0]`)

            note: `tiGridNode=None` if `mpds_simoutput->outputTiGridNodeIndexImage=NULL`

        - tiIndex : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `tiIndex[i]`: TI index map for the i-th realisation \
            (`mpds_simoutput->outputTiIndexImage[0]`)

            note: `tiIndex=None` if `mpds_simoutput->outputTiIndexImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'deesseRun'

    if not deesse_input.ok:
        err_msg = f'{fname}: check deesse input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f"{fname}: number of threads used will exceed number of cpu(s) of the system...")
        else:
            print(f"{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...")

    if deesse_input.mask is not None and add_data_point_to_mask and deesse_input.dataPointSet is not None:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(deesse_input.mask)
        # Add cell to mask if needed
        for ps in deesse_input.dataPointSet:
            im_tmp = img.imageFromPoints(ps.val[:3].T,
                    nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                    sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                    ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                    indicator_var=True,
                    logger=logger)
            deesse_input.mask = 1.0*np.any((im_tmp.val[0], deesse_input.mask), axis=0)
            del (im_tmp)

    if verbose > 1:
        if logger:
            logger.info(f"{fname}: DeeSse running... [" + \
                f"VERSION {deesse.MPDS_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
        else:
            print(f"{fname}: DeeSse running... [" + \
                f"VERSION {deesse.MPDS_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesse...

    # Convert deesse input from python to C
    try:
        mpds_siminput = deesse_input_py2C(deesse_input, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert deesse input from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # if mpds_siminput is None:
    #     err_msg = f'{fname}: cannot convert deesse input from python to C'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)

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
    err = deesse.MPDSOMPSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: deesse input
    deesse.MPDSFreeSimInput(mpds_siminput)
    deesse.free_MPDS_SIMINPUT(mpds_siminput)

    if err:
        # Free memory on C side: simulation output
        deesse.MPDSFreeSimOutput(mpds_simoutput)
        deesse.free_MPDS_SIMOUTPUT(mpds_simoutput)
        # Free memory on C side: progress monitor
        deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    deesse_output = deesse_output_C2py(mpds_simoutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: simulation output
    deesse.MPDSFreeSimOutput(mpds_simoutput)
    deesse.free_MPDS_SIMOUTPUT(mpds_simoutput)

    # Free memory on C side: progress monitor
    deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if deesse_input.mask is not None and add_data_point_to_mask and deesse_input.dataPointSet is not None:
        # Remove the value out of the original mask (using its copy see above)
        for im in deesse_output['sim']:
            im.val[:, mask_original==0.0] = np.nan

    if verbose > 1 and deesse_output:
        if logger:
            logger.info(f"{fname}: DeeSse run complete")
        else:
            print(f"{fname}: DeeSse run complete")

    # Show (print) encountered warnings
    if verbose > 0 and deesse_output and deesse_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({deesse_output['nwarning']} times in all):")
            for i, warning_message in enumerate(deesse_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return deesse_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseRun_mp(
        deesse_input,
        add_data_point_to_mask=True,
        nproc=-1,
        nthreads_per_proc=-1,
        verbose=2,
        logger=None):
    """
    Computes the same as the function :func:`deesseRun`, using multiprocessing.

    All the parameters are the same as those of the function :func:`deesseRun`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`deesseRun`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`deesseRun`.

    **Parameters (new)**
    --------------------
    nproc : int, default: -1
        number of process(es): a negative number (or zero), -n <= 0, can be specified 
        to use the total number of cpu(s) of the system except n; `nproc` is finally
        at maximum equal to `nreal` but at least 1 by applying:
        
        - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
        - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
        where nmax is the total number of cpu(s) of the system (retrieved by \
        `multiprocessing.cpu_count()`)

        note: if `nproc=None`, `nproc=-1` is used

    nthreads_per_proc : int, default: -1
        number of thread(s) per process;
        if `nthreads_per_proc = -n <= 0`: `nthreads_per_proc` is automatically 
        computed as the maximal integer (but at least 1) such that 
        `nproc*nthreads_per_proc <= nmax-n`, where nmax is the total number of cpu(s)
        of the system (retrieved by `multiprocessing.cpu_count()`); 

        note: if `nthreads_per_proc=None`, `nthreads_per_proc=-1` is used
    """
    fname = 'deesseRun_mp'

    if not deesse_input.ok:
        err_msg = f'{fname}: check deesse input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # if deesse_input.nrealization <= 1:
    #     if verbose > 1:
    #         if logger:
    #             logger.info(f'{fname}: number of realization does not exceed 1: launching deesseRun...')
    #         else:
    #             print(f'{fname}: number of realization does not exceed 1: launching deesseRun...')
    #     nthreads = nthreads_per_proc
    #     if nthreads is None:
    #         nthreads = -1
    #     deesse_output = deesseRun(deesse_input, add_data_point_to_mask=add_data_point_to_mask, nthreads=nthreads, verbose=verbose, logger=logger)
    #     return deesse_output

    # Set number of process(es): nproc
    if nproc is None:
        nproc = -1
    
    if nproc <= 0:
        nproc = max(min(multiprocessing.cpu_count() + nproc, deesse_input.nrealization), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), deesse_input.nrealization), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of processes has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of processes has been changed (now: nproc={nproc})')

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
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    if deesse_input.mask is not None and add_data_point_to_mask and deesse_input.dataPointSet is not None:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(deesse_input.mask)
        # Add cell to mask if needed
        for ps in deesse_input.dataPointSet:
            im_tmp = img.imageFromPoints(ps.val[:3].T,
                    nx=deesse_input.nx, ny=deesse_input.ny, nz=deesse_input.nz,
                    sx=deesse_input.sx, sy=deesse_input.sy, sz=deesse_input.sz,
                    ox=deesse_input.ox, oy=deesse_input.oy, oz=deesse_input.oz,
                    indicator_var=True,
                    logger=logger)
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

    if verbose > 1:
        if logger:
            logger.info(f"{fname}: DeeSse running on {nproc} process(es)... [" + \
                f"VERSION {deesse.MPDS_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
        else:
            print(f"{fname}: DeeSse running on {nproc} process(es)... [" + \
                f"VERSION {deesse.MPDS_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
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
        verb = 0
        # if i==0:
        #     verb = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verb = 0
        # Launch deesse (i-th process)
        out_pool.append(pool.apply_async(deesseRun, args=(input, False, nth, verb), kwds={'logger':logger}))

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
    nwarning = int(np.sum([out['nwarning'] for out in deesse_output_proc]))
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

    if deesse_input.mask is not None and add_data_point_to_mask and deesse_input.dataPointSet is not None:
        # Remove the value out of the original mask (using its copy see above)
        for im in sim:
            im.val[:, mask_original==0.0] = np.nan

    deesse_output = {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'sim_pyramid':sim_pyramid, 'sim_pyramid_var_original_index':sim_pyramid_var_original_index, 'sim_pyramid_var_pyramid_index':sim_pyramid_var_pyramid_index,
        'path':path, 'error':error, 'tiGridNode':tiGridNode, 'tiIndex':tiIndex,
        'nwarning':nwarning, 'warnings':warnings
        }

    if verbose > 1 and deesse_output:
        if logger:
            logger.info(f'{fname}: DeeSse run complete (all process(es))')
        else:
            print(f'{fname}: DeeSse run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and deesse_output and deesse_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({deesse_output['nwarning']} times in all):")
            for i, warning_message in enumerate(deesse_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return deesse_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def exportDeesseInput(
        deesse_input,
        dirname='input_ascii',
        fileprefix='ds',
        endofline='\n',
        verbose=1,
        logger=None):
    """
    Exports deesse input in txt (ASCII) files (in the directory `dirname`).

    The command line version of deesse can then be launched from the directory
    `dirname` by using the generated txt files.

    Parameters
    ----------
    deesse_input : :class:`DeesseInput`
        deesse input in python

    dirname : str, default: 'input_ascii'
        name of the directory in which the files will be written;
        if not existing, it will be created;
        WARNING: the generated files might erase already existing ones!

    fileprefix : str, default: 'ds'
        prefix for generated files, the main input file will be
        "`dirname`/`fileprefix`.in"

    endofline : str, default: '\\\\n'
        end of line character

    verbose : int, default: 1
        verbose mode for comments in the written main input file:

        - 0: no comment
        - 1: basic comments
        - 2: detailed comments

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    fname = 'exportDeesseInput'

    if not deesse_input.ok:
        err_msg = f'{fname}: check deesse input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Create ouptut directory if needed
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Convert deesse input from python to C
    try:
        mpds_siminput = deesse_input_py2C(deesse_input, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert deesse input from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # if mpds_siminput is None:
    #     err_msg = f'{fname}: cannot convert deesse input from python to C'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)

    err = deesse.MPDSExportSimInput(mpds_siminput, dirname, fileprefix, endofline, verbose)

    if err:
        # Free memory on C side: deesse input
        deesse.MPDSFreeSimInput(mpds_siminput)
        deesse.free_MPDS_SIMINPUT(mpds_siminput)
        # Raise error
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Free memory on C side: deesse input
    deesse.MPDSFreeSimInput(mpds_siminput)
    deesse.free_MPDS_SIMINPUT(mpds_siminput)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def importDeesseInput(filename, dirname='.', logger=None):
    """
    Imports deesse input from txt (ASCII) files.

    The files used for command line version of deesse (from the directory named
    `dirname`) are read and the corresponding deesse input in python is
    returned.

    Parameters
    ----------
    filename : str
        name of the main input txt (ASCII) file (without path) used for the
        command line version of deesse

    dirname : str, default: '.'
        name of the directory in which the main input txt (ASCII) file is stored
        (and from which the command line version of deesse would be launched)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesse_input :class:`DeesseInput`
        deesse input in python
    """
    fname = 'importDeesseInput'

    # Check directory
    if not os.path.isdir(dirname):
        err_msg = f'{fname}: directory does not exist'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Check file
    if not os.path.isfile(os.path.join(dirname, filename)):
        err_msg = f'{fname}: input file does not exist'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        deesse_input = deesse_input_C2py(mpds_siminput, logger=logger)

    except:
        # Free memory on C side: deesse input
        deesse.delete_MPDS_SIMINPUTp(mpds_siminputp)
        # Raise error
        err_msg = f'{fname}: cannot import deesse input from ASCII files'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    finally:
        # Change directory (to initial working directory)
        os.chdir(cwd)

    # Free memory on C side: deesse input
    deesse.delete_MPDS_SIMINPUTp(mpds_siminputp)

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
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes the Gaussian (pyramid) reduction or expansion of the input image.

    This function applies the Gaussian pyramid reduction or expansion to all
    variables (treated as continuous) of the input image, and returns an output
    image with the same number of variables, whose the names are the same as the
    variables of the input image, followed by a suffix the suffix "\\_GPred" (resp.
    "\\_GPexp") if reduction (resp. expansion) is applied. The grid (support) of
    the output image is derived from the Gaussian pyramid operation.
    The Gaussian operation consists in applying a weighted moving average using a
    Gaussian-like kernel (or filter) of size (2*kx + 1) x (2*ky + 1) x (2*kz + 1)
    [see parameters below], while in the output image grid the number of cells
    along x, y, z-axis will be divided (resp. multiplied) by a factor (of about)
    kx, ky, kz respectively if reduction (resp. expansion) is applied.

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    operation : str {'reduce', 'expand'}, default: 'reduce'
        operation to apply

    kx : int, optional
        reduction step along x axis

        * `kx = 0`: nothing is done, same dimension in output
        * `kx = 1`: same dimension in output (with weighted average over 3 nodes)
        * `kx = 2`: classical gaussian pyramid
        * `kx > 2`: generalized gaussian pyramid

        by defaut (`None`): the reduction step `kx` is set to 2 if the input image
        grid has more than one cell along x axis, and to 0 otherwise

    ky : int, optional
        reduction step along y axis; see `kx` for details

    kz : int, optional
        reduction step along z axis; see `kx` for details

    w0x : float, optional
        weight for central cell in the kernel (filter) when computing average
        during Gaussian pyramid operation, along x axis;
        specifying a positive value or zero implies to explicitly set the weight;
        by default (`None`) or if a negative value is set: the default weight
        derived from proportionality with Gaussian weights (binomial
        coefficients) will be used

    w0y : float, optional
        weight for central cell in the kernel (filter) when computing average
        during Gaussian pyramid operation, along y axis; see `w0x` for details

    w0z : float, optional
        weight for central cell in the kernel (filter) when computing average
        during Gaussian pyramid operation, along z axis; see `w0x` for details

    minWeight : float, optional
        minimal weight on informed cells within the filter to define output
        value: when applying the moving weighted average, if the total weight
        on informed cells within the kernel (filter) is less than `minWeight`,
        undefined value (`numpy.nan`) is set as output value, otherwise the
        weighted average is set;
        specifying a positive value or zero implies to explicitly set the
        minimal weight

        By default (`None`) or if a negative value is set: a default minimal
        weight will be used

        Note: the default minimal weight is
        `geone.deesse_core.deesse.MPDS_GAUSSIAN_PYRAMID_RED_TOTAL_WEIGHT_MIN`
        for reduction, and
        `geone.deesse_core.deesse.MPDS_GAUSSIAN_PYRAMID_EXP_TOTAL_WEIGHT_MIN`
        for expansion

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
        output image (see above)
    """
    fname = 'imgPyramidImage'

    # Check
    if operation not in ('reduce', 'expand'):
        err_msg = f'{fname}: unknown `operation`'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Prepare parameters
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
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # Allocate output image "in C"
    output_image_c = deesse.malloc_MPDS_IMAGE()
    deesse.MPDSInitImage(output_image_c)

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

    # Compute pyramid (launch C code)
    if operation == 'reduce':
        err = deesse.MPDSOMPImagePyramidReduce(input_image_c, output_image_c, kx, ky, kz, w0x, w0y, w0z, minWeight, nth)
    elif operation == 'expand':
        err = deesse.MPDSOMPImagePyramidExpand(input_image_c, output_image_c, kx, ky, kz, w0x, w0y, w0z, minWeight, nth)
    else:
        # Free memory on C side: input_image_c
        deesse.MPDSFreeImage(input_image_c)
        deesse.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        deesse.MPDSFreeImage(output_image_c)
        deesse.free_MPDS_IMAGE(output_image_c)
        # Raise error
        err_msg = f'{fname}: `operation` invalid'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Retrieve output image "in python"
    if err:
        # Free memory on C side: input_image_c
        deesse.MPDSFreeImage(input_image_c)
        deesse.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        deesse.MPDSFreeImage(output_image_c)
        deesse.free_MPDS_IMAGE(output_image_c)
        # Raise error
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    output_image = img_C2py(output_image_c, logger=logger)

    # Free memory on C side: input_image_c
    deesse.MPDSFreeImage(input_image_c)
    deesse.free_MPDS_IMAGE(input_image_c)
    # Free memory on C side: output_image_c
    deesse.MPDSFreeImage(output_image_c)
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
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Transforms variable(s) of an image from "categorical" to "continuous".

    Transforms the desired variable(s), considered as "categorical", from the
    input image, into "continuous" variable(s) (with values in [0, 1]), and
    returns the corresponding output image. The transformation for a variable
    with n categories is done such that:

    - each category in input will correspond to a distinct output value \
    in {i/(n-1), i=0, ..., n-1}
    - the output values are set such that "closer values correspond to better \
    connected (more contact btw.) categories"
    - this is the transformation done by deesse when pyramid is used with \
    pyramid type ('pyramidType') set to 'categorical_to_continuous'

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    varInd : sequence of ints, or int, optional
        index(es) of the variables for which the transformation has to be done;
        by default: all variables are transformed

    xConnectFlag : bool, optional
        flag indicating if the connction (contact btw.) categories are
        taken into account along x axis (`True`) or not (`False`);
        by default (`None`): set to `True` provided that the number of cells of
        the input image along x axis is greater than 1 (set to `False` otherwise)

    yConnectFlag : bool, optional
        as `xConnectFlag`, but for y axis

    zConnectFlag : bool, optional
        as `xConnectFlag`, but for z axis

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
        output image (see above)
    """
    fname = 'imgCategoricalToContinuous'

    # Check
    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if np.sum([iv in range(input_image.nv) for iv in varInd]) != len(varInd):
            err_msg = f'{fname}: invalid index-es'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

    else:
        varInd = np.arange(input_image.nv)

    # Prepare parameters
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
    output_image = img.copyImg(input_image, logger=logger)

    # Initialize value index
    val_index = np.zeros(input_image.nxyz(), dtype='intc')

    # Initialize vector in C
    val_index_c = deesse.new_int_array(int(val_index.size))

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

    for ind in varInd:
        # Get vector of values of the variale of index ind from input image
        vptr = input_image.val[ind].reshape(-1)

        unique_val = input_image.get_unique_one_var(ind, logger=logger)
        for i, v in enumerate(unique_val):
            val_index[vptr==v] = i

        # ... set index -1 for nan
        val_index[np.isnan(vptr)] = -1

        n = len(unique_val)

        # Set vectors in C
        deesse.mpds_set_int_vector_from_array(val_index_c, 0, val_index)

        to_new_index_c = deesse.new_int_array(n)
        to_initial_index_c = deesse.new_int_array(n)

        # Compute index correspondence (launch C code)
        err = deesse.MPDSOMPGetImageOneVarNewValueIndexOrder(
                input_image.nx, input_image.ny, input_image.nz,
                n, val_index_c,
                to_new_index_c, to_initial_index_c,
                int(xConnectFlag), int(yConnectFlag), int(zConnectFlag),
                nth)

        # Retrieve vector to_new_index and to_initial_index "in python"
        if err:
            # Free memory on C side
            deesse.delete_int_array(to_new_index_c)
            deesse.delete_int_array(to_initial_index_c)
            # Free memory on C side
            deesse.delete_int_array(val_index_c)
            # Raise error
            err_message = deesse.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            err_msg = f'{fname}: {err_message}'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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

    # Free memory on C side
    deesse.delete_int_array(val_index_c)

    return output_image
# ----------------------------------------------------------------------------

##### Additional stuff for deesseX #####

# ============================================================================
class DeesseXInputSectionPath(object):
    """
    Class defining main input parameters for cross-simulation (deesseX).

    **Attributes**

    sectionMode: str {'section_xy_xz_yz', \
                      'section_xy_yz_xz', \
                      'section_xz_xy_yz', \
                      'section_xz_yz_xy', \
                      'section_yz_xy_xz', \
                      'section_yz_xz_xy', \
                      'section_xy_xz', \
                      'section_xz_xy', \
                      'section_xy_yz', \
                      'section_yz_xy', \
                      'section_xz_yz', \
                      'section_yz_xz', \
                      'section_xy_z', \
                      'section_z_xy', \
                      'section_xz_y', \
                      'section_y_xz', \
                      'section_yz_x', \
                      'section_x_yz', \
                      'section_x_y_z', \
                      'section_x_z_y', \
                      'section_y_x_z', \
                      'section_y_z_x', \
                      'section_z_x_y', \
                      'section_z_y_x', \
                      'section_x_y', \
                      'section_y_x', \
                      'section_x_z', \
                      'section_z_x', \
                      'section_y_z', \
                      'section_z_y'}, default: 'section_xz_yz'
        section mode, defining which type of section will be simulated
        alternately: 'section_<t_1>_<t_2>[_<t_3>]': means that simulation in 2D
        (resp. 1D) will be done alternately in sections parallel to the plane
        (resp. axis) given in the string '<t_i>'

        Notes:

        - the order can matter (depending on `sectionPathMode`)
        - the mode involving only two 1D axis as section (i.e. 'section_x_y' to \
        'section_z_y') can be used for a two-dimensional simulation grid

    sectionPathMode : str {'section_path_random', 'section_path_pow_2', \
                        'section_path_subdiv', 'section_path_manual'}, \
                        default: 'section_path_subdiv'
        section path mode, defining the section path, i.e. the succession of
        simulated sections:

        - 'section_path_random': random section path
        - 'section_path_pow_2': indexes (of cells locating the section) \
        in the orthogonal direction of the sections, are chosen as \
        decreasing power of 2 (dealing alternately with each section \
        orientation in the order given by `sectionMode`)
        - 'section_path_subdiv': succession of sections is defined as:
            (a) for each section orientation (in the order given by \
            `sectionMode`), the section corresponding to the most left \
            border (containing the origin) of the simulation grid is \
            selected
            (b) let minspaceX, minspaceY, minspaceZ (see parameters \
            below), the minimal space (or step) in number of cells \
            along x, y, z axis resp. between two successive sections \
            of the same type and orthogonal to x, y, z axis resp.:

                - \\(i\\) for each section orientation (in the order given by \
                `sectionMode`): the section(s) corresponding to the \
                most right border (face or edge located at one largest \
                index in the corresponding direction) of the \
                simulation grid is selected, provided that the space \
                (step) with the previous section (selected in (a)) \
                satisfies the minimal space in the relevant direction
                - \\(ii\\) for each section orientation (in the order given by \
                `sectionMode`): the sections between the borders are \
                selected, such that they are regularly spaced along \
                any direction (with a difference of at most one cell) \
                and such that the minimal space is satisfied (i.e. \
                the number of cell from one section to the next one \
                is at least equal to corresponding parameter \
                minspaceX, minspaceY or minspaceZ)
                - \\(iii\\) for each section orientation (in the order given by \
                `sectionMode`): if in step (i) the right border was not \
                selected (due to a space less than the minimal space \
                paremeter(s)), then it is selected here

                note that at the end of step (b), there are at least two \
                sections of same type along any axis direction (having \
                more than one cell in the entire simulation grid)

            (c) next, the middle sections (along each direction) between \
            each pair of consecutive sections already selected are \
            selected, until the entire simulation grid is filled, \
            following one of the two methods (see parameter \
            `balancedFillingFlag`):

                - if `balancedFillingFlag=False`: \
                considering alternately each section orientation, in \
                the order given by `sectionMode`,
                - if `balancedFillingFlag=True`: \
                choosing the axis direction (x, y, or z) for which \
                the space (in number of cells) between two \
                consecutive sections already selected is the largest, \
                then selecting the section orientation(s) (among \
                those given by `sectionMode`) orthogonal to that \
                direction, and considering the middle sections with \
                respect to that direction

        - 'section_path_manual': succession of sections explicitly \
        given (see parameters `nsection`, `sectionType` and \
        `sectionLoc`)

    minSpaceX : float, optional
        used iff `sectionPathMode='section_path_subdiv'`,
        minimal space in number of cells along x direction, in step (b) above;
        note:

        - if `minSpaceX > 0`: use as it in step (b)
        - if `minSpaceX = 0`: ignore (skip) step (b,ii) for x direction
        - if `minSpaceX < 0`: this parameter is automatically computed,

        and defined as the "range" in the x direction computed
        from the training image(s) used in section(s) including
        the x direction

    minSpaceY : float, optional
        same as `minSpaceX`, but in y direction

    minSpaceZ : float, optional
        same as `minSpaceX`, but in z direction

    balancedFillingFlag : bool, default: True
        used iff `sectionPathMode='section_path_subdiv'`,
        flag defining the method used in step (c) above

    nsection : int, default: 0
        used iff `sectionPathMode='section_path_manual'`,
        number of section(s) to be simulated at total [sections (2D and/or 1D)];
        note: a partial filling of the simulation grid can be considered

    sectionType : sequence of ints of length `nsection`, optional
        used iff `sectionPathMode='section_path_manual'`,
        indexes identifying the type of sections:

        - `sectionType[i]`: type id of the i-th simulated section, \
        for 0 <= i < `nsection`, with:
            * id = 0: xy section (2D)
            * id = 1: xz section (2D)
            * id = 2: yz section (2D)
            * id = 3: z section (1D)
            * id = 4: y section (1D)
            * id = 5: x section (1D)

    sectionLoc : sequence of ints of length `nsection`, optional
        used iff `sectionPathMode='section_path_manual'`,
        indexes location of sections:

        - `sectionLoc[i]`: location of the i-th simulated section, \
        for 0 <= i < `nsection`, with:
            * if sectionType[i] = 0 (xy), then \
            sectionLoc[i]=k in {0, ..., nz-1}, \
            k is the index location along x axis
            * if sectionType[i] = 1 (xz), then \
            sectionLoc[i]=k in {0, ..., ny-1}, \
            k is the index location along y axis
            * if sectionType[i] = 2 (yz), then \
            sectionLoc[i]=k in {0, ..., nx-1}, \
            k is the index location along z axis
            * if sectionType[i] = 3 (z), then \
            sectionLoc[i]=k in {0, ..., nx*ny-1}, \
            (k%nx, k//nx) is the two index locations in xy section
            * if sectionType[i] = 4 (y), then \
            sectionLoc[i]=k in {0, ..., nx*nz-1}, \
            (k%nx, k//nx) is the two index locations in xz section
            * if sectionType[i] = 5 (x), then \
            sectionLoc[i]=k in {0, ..., ny*nz-1}, \
            (k%ny, k//ny) is the two index locations in yz section

            where nx, ny, nz are the number of nodes in the entire
            simulation grid along x, y, z axis respectively

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
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
                 sectionLoc=None,
                 logger=None):
        """
        Inits an instance of the class.
        """
        fname = 'DeesseXInputSectionPath'

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

        if sectionMode not in sectionMode_avail:
            err_msg = f'{fname}: unknown `sectionMode`'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.sectionMode = sectionMode

        # sectionPathMode
        sectionPathMode_avail = (
            'section_path_random',
            'section_path_pow_2',
            'section_path_subdiv',
            'section_path_manual'
        )
        if sectionPathMode not in sectionPathMode_avail:
            err_msg = f'{fname}: unknown `sectionPathMode`'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `sectionType`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

                try:
                    self.sectionLoc = np.asarray(sectionLoc, dtype='int').reshape(nsection)
                except:
                    err_msg = f'{fname}: parameter `sectionLoc`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
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
    Class input parameters for one section type (deesseX).

    **Attributes**

    nx : int, default: 0
        number of cells along x axis in the entire simulation grid (SG);
        should be consistent with the "parent" class :class:`DeesseXInput`
        (i.e. defined as in the "parent" class :class:`DeesseXInput`)

    ny : int, default: 0
        same as `nx`, but for y axis

    ny : int, default: 0
        same as `nx`, but for z axis

    nv : int, default: 0
        number of variable(s) / attribute(s);
        should be consistent with the "parent" class :class:`DeesseXInput`
        (defined as in the "parent" class :class:`DeesseXInput`)

    distanceType : [sequence of] int(s) or str(s), optional
        type of distance (between pattern) for each variable
        (defined as in the "parent" class :class:`DeesseXInput`)

    sectionType : str or int, optional
        type of section, possible values:

        - 'xy' or 'XY' or 0: 2D section parallel to the plane xy
        - 'xz' or 'XZ' or 1: 2D section parallel to the plane xz
        - 'yz' or 'YZ' or 2: 2D section parallel to the plane yz
        - 'z' or 'Z' or 3:   1D section parallel to the axis z
        - 'y' or 'Y' or 4:   1D section parallel to the axis y
        - 'x' or 'X' or 5:   1D section parallel to the axis x

    nTI : int, optional
        as in :class:`DeesseInput`

    TI : [sequence of] :class:`geone.img.Img`
        as in :class:`DeesseInput`

    simGridAsTiFlag : [sequence of] bool(s), optional
        as in :class:`DeesseInput`

    pdfTI : array-like of floats, optional
        as in :class:`DeesseInput`

    homothetyUsage : int, default: 0
        as in :class:`DeesseInput`

    homothetyXLocal : bool, default: False
        as in :class:`DeesseInput`

    homothetyXRatio: array-like of floats, or float, optional
        as in :class:`DeesseInput`;
        note: if given "locally", the dimension of the entire SG is considered

    homothetyYLocal, homothetyYRatio :
        as in :class:`DeesseInput`

    homothetyZLocal, homothetyZRatio :
        as in :class:`DeesseInput`

    rotationUsage : int, default: 0
        as in :class:`DeesseInput`

    rotationAzimuthLocal : bool, default: False
        as in :class:`DeesseInput`;
        note: if given "locally", the dimension of the entire SG is considered

    rotationAzimuth : array-like of floats, or float, optional
        as in :class:`DeesseInput`

    rotationDipLocal, rotationDip :
        as in :class:`DeesseInput`

    rotationPlungeLocal, rotationPlunge :
        as in :class:`DeesseInput`

    searchNeighborhoodParameters : [sequence of] :class:`SearchNeighborhoodParameters`, optional
        as in :class:`DeesseInput`

    nneighboringNode : [sequence of] int(s), optional
        as in :class:`DeesseInput`

    maxPropInequalityNode : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    neighboringNodeDensity : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    simType : str {'sim_one_by_one', 'sim_variable_vector'}, default: 'sim_one_by_one'
        as in :class:`DeesseInput`;
        note: defines the type of simulation within the section

    simPathType : str {'random', \
                    'random_hd_distance_pdf', 'random_hd_distance_sort', \
                    'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort', \
                    'unilateral'}, default: 'random'
        as in :class:`DeesseInput`;
        note: defines the type of path within the section

    simPathStrength : double, optional
        as in :class:`DeesseInput`;
        note: defines the type of path within the section

    simPathPower : double, optional
        as in :class:`DeesseInput`;
        note: defines the type of path within the section

    simPathUnilateralOrder : sequence of ints, optional
        as in :class:`DeesseInput`;
        note: defines the type of path within the section

    distanceThreshold : [sequence of] float(s), optional
        as in :class:`DeesseInput`

    softProbability : [sequence of] :class:`SoftProbability`, optional
        as in :class:`DeesseInput`

    maxScanFraction : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    pyramidGeneralParameters : :class:`PyramidGeneralParameters`, optional
        as in :class:`DeesseInput`;
        note: defining the general pyramid parameters for the simulation within
        the section

    pyramidParameters : [sequence of] :class:`PyramidParameters`, optional
        as in :class:`DeesseInput`;
        note: defining the pyramid parameters for the simulation within
        the section

    tolerance : float, default: 0.0
        as in :class:`DeesseInput`

    npostProcessingPathMax : int, default: 0
        as in :class:`DeesseInput`

    postProcessingNneighboringNode : [sequence of] int(s), optional
        as in :class:`DeesseInput`

    postProcessingNeighboringNodeDensity : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    postProcessingDistanceThreshold : [sequence of] float(s), optional
        as in :class:`DeesseInput`

    postProcessingMaxScanFraction : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    postProcessingTolerance : float, default: 0.0
        as in :class:`DeesseInput`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    **Methods**
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
                 postProcessingTolerance=0.0,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
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
                            err_msg = f'{fname}: parameter `distanceType`...'
                            if logger: logger.error(err_msg)
                            raise DeesseinterfaceError(err_msg)

                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `distanceType`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # dimension
        dim = int(nx>1) + int(ny>1) + int(nz>1)

        # section type
        if sectionType is None:
            err_msg = f'{fname}: parameter `sectionType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `sectionType`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        elif isinstance(sectionType, int):
            self.sectionType = sectionType

        else:
            err_msg = f'{fname}: parameter `sectionType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        # TI, simGridAsTiFlag, nTI
        if TI is None and simGridAsTiFlag is None:
            err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid (both `None`)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if TI is not None:
            self.TI = np.asarray(TI).reshape(-1)

        if simGridAsTiFlag is not None:
            self.simGridAsTiFlag = np.asarray(simGridAsTiFlag, dtype='bool').reshape(-1)

        if TI is None:
            self.TI = np.array([None for i in range(len(self.simGridAsTiFlag))], dtype=object)

        if simGridAsTiFlag is None:
            self.simGridAsTiFlag = np.array([False for i in range(len(self.TI))], dtype='bool') # set dtype='bool' in case of len(self.TI)=0

        if len(self.TI) != len(self.simGridAsTiFlag):
            err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid (not same length)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        for f, t in zip(self.simGridAsTiFlag, self.TI):
            if (not f and t is None) or (f and t is not None):
                err_msg = f'{fname}: `TI` / `simGridAsTiFlag` invalid...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if nTI is not None and nTI != len(self.TI):
            err_msg = f'{fname}: `nTI` invalid...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `pdfTI`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

        # homothety
        if homothetyUsage == 1:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif homothetyUsage == 2:
            if homothetyXLocal:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyXRatio is None:
                    self.homothetyXRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyXRatio = np.asarray(homothetyXRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyXRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyYLocal:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyYRatio is None:
                    self.homothetyYRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyYRatio = np.asarray(homothetyYRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyYRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if homothetyZLocal:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.repeat(1., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if homothetyZRatio is None:
                    self.homothetyZRatio = np.array([1., 1.])
                else:
                    try:
                        self.homothetyZRatio = np.asarray(homothetyZRatio, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `homothetyZRatio`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif homothetyUsage == 0:
            self.homothetyXRatio = None
            self.homothetyYRatio = None
            self.homothetyZRatio = None

        else:
            err_msg = f'{fname}: `homothetyUsage` invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., nx*ny*nz).reshape(nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(1)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif rotationUsage == 2:
            if rotationAzimuthLocal:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationAzimuth is None:
                    self.rotationAzimuth = np.array([0., 0.])
                else:
                    try:
                        self.rotationAzimuth = np.asarray(rotationAzimuth, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationAzimuth`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationDipLocal:
                if rotationDip is None:
                    self.rotationDip = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationDip is None:
                    self.rotationDip = np.array([0., 0.])
                else:
                    try:
                        self.rotationDip = np.asarray(rotationDip, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationDip`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            if rotationPlungeLocal:
                if rotationPlunge is None:
                    self.rotationPlunge = np.repeat(0., 2*nx*ny*nz).reshape(2, nz, ny, nx)
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2, nz, ny, nx)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

            else:
                if rotationPlunge is None:
                    self.rotationPlunge = np.array([0., 0.])
                else:
                    try:
                        self.rotationPlunge = np.asarray(rotationPlunge, dtype=float).reshape(2)
                    except:
                        err_msg = f'{fname}: parameter `rotationPlunge`...'
                        if logger: logger.error(err_msg)
                        raise DeesseinterfaceError(err_msg)

        elif rotationUsage == 0:
            self.rotationAzimuth = None
            self.rotationDip = None
            self.rotationPlunge = None

        else:
            err_msg = f'{fname}: `rotationUsage` invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `searchNeighborhoodParameters`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `nneighboringNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if maxPropInequalityNode is None:
            self.maxPropInequalityNode = np.array([0.25 for i in range(nv)])
        else:
            try:
                self.maxPropInequalityNode = np.asarray(maxPropInequalityNode).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `maxPropInequalityNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if neighboringNodeDensity is None:
            self.neighboringNodeDensity = np.array([1. for i in range(nv)])
        else:
            try:
                self.neighboringNodeDensity = np.asarray(neighboringNodeDensity, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `neighboringNodeDensity`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # simulation type and simulation path type
        if simType not in ('sim_one_by_one', 'sim_variable_vector'):
            err_msg = f'{fname}: parameter `simType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.simType = simType

        if simPathType not in (
                'random',
                'random_hd_distance_pdf', 'random_hd_distance_sort',
                'random_hd_distance_sum_pdf', 'random_hd_distance_sum_sort',
                'unilateral'):
            err_msg = f'{fname}: parameter `simPathType`...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
                    err_msg = f'{fname}: parameter `simPathUnilateralOrder`...'
                    if logger: logger.error(err_msg)
                    raise DeesseinterfaceError(err_msg)

        else:
            self.simPathUnilateralOrder = None

        # distance threshold
        if distanceThreshold is None:
            self.distanceThreshold = np.array([0.05 for i in range(nv)])
        else:
            try:
                self.distanceThreshold = np.asarray(distanceThreshold, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `distanceThreshold`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # soft probability
        if softProbability is None:
            self.softProbability = np.array([SoftProbability(probabilityConstraintUsage=0, logger=logger) for i in range(nv)])
        else:
            try:
                self.softProbability = np.asarray(softProbability).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `softProbability`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `maxScanFraction`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # pyramids
        if pyramidGeneralParameters is None:
            self.pyramidGeneralParameters = PyramidGeneralParameters(nx=nx, ny=ny, nz=nz, logger=logger)
        else:
            self.pyramidGeneralParameters = pyramidGeneralParameters

        if pyramidParameters is None:
            self.pyramidParameters = np.array([PyramidParameters() for _ in range(nv)])
        else:
            try:
                self.pyramidParameters = np.asarray(pyramidParameters).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `pyramidParameters`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingNneighboringNode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingNeighboringNodeDensity`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `postProcessingDistanceThreshold`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if postProcessingMaxScanFraction is None:
            self.postProcessingMaxScanFraction = np.array([min(deesse.MPDS_POST_PROCESSING_MAX_SCAN_FRACTION_DEFAULT, self.maxScanFraction[i]) for i in range(nTI)], dtype=float)

        else:
            try:
                self.postProcessingMaxScanFraction = np.asarray(postProcessingMaxScanFraction, dtype=float).reshape(nTI)
            except:
                err_msg = f'{fname}: parameter `postProcessingMaxScanFraction`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        self.postProcessingTolerance = postProcessingTolerance

        self.ok = True # flag to "validate" the class

    # ------------------------------------------------------------------------
    # def __str__(self):
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** DeesseXInputSection object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ============================================================================
class DeesseXInput(object):
    """
    Class defining main input parameters for deesseX (cross-simulation/X-simulation).

    **Attributes**

    simName : str, default: 'deesseX_py'
        simulation name (useless)

    nx : int, default: 0
        as in :class:`DeesseInput`

    ny : int, default: 0
        as in :class:`DeesseInput`

    nz : int, default: 0
        as in :class:`DeesseInput`

    sx : float, default: 1.0
        as in :class:`DeesseInput`

    sy : float, default: 1.0
        as in :class:`DeesseInput`

    sz : float, default: 1.0
        as in :class:`DeesseInput`

    ox : float, default: 1.0
        as in :class:`DeesseInput`

    oy : float, default: 1.0
        as in :class:`DeesseInput`

    oz : float, default: 1.0
        as in :class:`DeesseInput`

    nv : int, default: 0
        as in :class:`DeesseInput`

    varname : sequence of strs of length `nv`, optional
        as in :class:`DeesseInput`

    outputVarFlag : sequence of bools of length `nv`, optional
        as in :class:`DeesseInput`

    outputSectionTypeFlag : bool, default: False
        indicates if "section type" map(s) is (are) retrieved in output;
        one file per realization if `sectionPathMode='section_path_random'`,
        and one file in all otherwise (same for each realization);
        "section type" is an index identifiying the type of section;
        see `sectionPath_parameters` (below) and
        `sectionPathMode` and `sectionType` in :class:`DeesseXInputSectionPath`

    outputSectionStepFlag : bool, default: False
        indicates if "section step" map(s) is (are) retrieved in output;
        one file per realization if `sectionPathMode='section_path_random'`,
        and one file in all otherwise (same for each realization;
        "section step" is the step index of simulation by deesse of (a bunch of)
        sections of same type;
        see `sectionPath_parameters` (below) and
        `sectionPathMode` in :class:`DeesseXInputSectionPath`

    outputReportFlag : bool, default: False
        as in :class:`DeesseInput`

    outputReportFileName : str, optional
        as in :class:`DeesseInput`

    dataImage : sequence of :class:`geone.img.Img`, optional
        as in :class:`DeesseInput`

    dataPointSet : sequence of :class:`geone.img.PointSet`, optional
        as in :class:`DeesseInput`

    mask : array-like, optional
        as in :class:`DeesseInput`

    expMax : float, default: 0.05
        as in :class:`DeesseInput`

    normalizingType : str {'linear', 'uniform', 'normal'}, default: 'linear'
        as in :class:`DeesseInput`

    rescalingMode : [sequence of] str(s), optional
        as in :class:`DeesseInput`

    rescalingTargetMin : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    rescalingTargetMax : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    rescalingTargetMean : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    rescalingTargetLength : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    relativeDistanceFlag : [sequence of] bool(s), optional
        as in :class:`DeesseInput`

    distanceType : [sequence of] int(s) or str(s), optional
        as in :class:`DeesseInput`

    powerLpDistance : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    powerLpDistanceInv : [sequence of] double(s), optional
        as in :class:`DeesseInput`

    conditioningWeightFactor : [sequence of] float(s), optional
        as in :class:`DeesseInput`

    sectionPath_parameters : :class:`DeesseXInputSectionPath`
       defines the overall strategy of simulation as a succession of section

    section_parameters : sequence of :class:`DeesseXInputSection`
        each element defines the parameter for one section type

    seed : int, default: 1234
        as in :class:`DeesseInput`

    seedIncrement : int, default: 1
        as in :class:`DeesseInput`

    nrealization : int, default: 1
        as in :class:`DeesseInput`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    In output simulated images (obtained by running DeeSseX), the names of the
    output variables are set to '<vname>_real<n>', where

    - <vname> is the name of the variable,
    - <n> is the realization index (starting from 0) \
    [<n> is written on 5 digits, with leading zeros]

    **Methods**
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
                 nrealization=1,
                 logger=None):
        """
        Inits an instance of the class.

        **Parameters** : see "Attributes" in the class definition above.
        """
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
            self.varname = ['V{i:d}' for i in range(nv)]
        else:
            try:
                self.varname = list(np.asarray(varname).reshape(nv))
            except:
                err_msg = f'{fname}: parameter `varname`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # outputVarFlag
        if outputVarFlag is None:
            self.outputVarFlag = np.array([True for i in range(nv)], dtype='bool')
        else:
            try:
                self.outputVarFlag = np.asarray(outputVarFlag, dtype='bool').reshape(nv)
            except:
                err_msg = f'{fname}: parameter `outputVarFlag`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `mask`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                err_msg = f'{fname}: parameter `rescalingMode`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMin is None:
            self.rescalingTargetMin = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMin = np.asarray(rescalingTargetMin, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMin`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMax is None:
            self.rescalingTargetMax = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMax = np.asarray(rescalingTargetMax, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMax`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetMean is None:
            self.rescalingTargetMean = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetMean = np.asarray(rescalingTargetMean, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetMean`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if rescalingTargetLength is None:
            self.rescalingTargetLength = np.array([0.0 for i in range(nv)], dtype=float)
        else:
            try:
                self.rescalingTargetLength = np.asarray(rescalingTargetLength, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `rescalingTargetLength`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # distance, ...
        if relativeDistanceFlag is None:
            self.relativeDistanceFlag = np.array([False for i in range(nv)], dtype='bool') # set dtype='bool' in case of nv=0
        else:
            try:
                self.relativeDistanceFlag = np.asarray(relativeDistanceFlag, dtype='bool').reshape(nv)
            except:
                err_msg = f'{fname}: parameter `relativeDistanceFlag`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        if powerLpDistance is None:
            self.powerLpDistance = np.array([1. for i in range(nv)])
        else:
            try:
                self.powerLpDistance = np.asarray(powerLpDistance, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `powerLpDistance`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

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
                            err_msg = f'{fname}: parameter `distanceType`...'
                            if logger: logger.error(err_msg)
                            raise DeesseinterfaceError(err_msg)

                self.distanceType = np.asarray(self.distanceType).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `distanceType`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # conditioning weight
        if conditioningWeightFactor is None:
            self.conditioningWeightFactor = np.array([1. for i in range(nv)])
        else:
            try:
                self.conditioningWeightFactor = np.asarray(conditioningWeightFactor, dtype=float).reshape(nv)
            except:
                err_msg = f'{fname}: parameter `conditioningWeightFactor`...'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        # sectionPath_parameters
        if sectionPath_parameters is None:
            err_msg = f'{fname}: parameter `sectionPath_parameters` (must be specified)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        self.sectionPath_parameters = sectionPath_parameters

        # section_parameters
        if section_parameters is None:
            err_msg = f'{fname}: parameter `section_parameters` (must be specified)...'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** DeesseInput object ***'
        out = out + '\n' + "use '.__dict__' to print details"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ----------------------------------------------------------------------------
def deesseX_input_sectionPath_py2C(sectionPath_parameters, logger=None):
    """
    Converts section path parameters (deesseX) from python to C
    (MPDS_XSECTIONPARAMETERS).

    Parameters
    ----------
    sectionPath_parameters : :class:`DeesseXInputSectionPath`
        section path parameters (strategy of simulation) in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_xsectionParameters : \\(MPDS_XSECTIONPARAMETERS \\*\\)
        section path parameters (strategy of simulation) in C
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
        # Free memory on C side
        deesse.MPDSFreeXSectionParameters(mpds_xsectionParameters)
        deesse.free_MPDS_XSECTIONPARAMETERS(mpds_xsectionParameters)
        # Raise error
        err_msg = f'{fname}: section mode unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        # Free memory on C side
        deesse.MPDSFreeXSectionParameters(mpds_xsectionParameters)
        deesse.free_MPDS_XSECTIONPARAMETERS(mpds_xsectionParameters)
        # Raise error
        err_msg = f'{fname}: section path type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    return mpds_xsectionParameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_sectionPath_C2py(mpds_xsectionParameters, logger=None):
    """
    Converts section path parameters (deesseX) from C to python.

    Parameters
    ----------
    mpds_xsectionParameters : \\(MPDS_XSECTIONPARAMETERS \\*\\)
        section path parameters (strategy of simulation) in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sectionPath_parameters : :class:`DeesseXInputSectionPath`
        section path parameters (strategy of simulation) in python
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
        err_msg = f'{fname}: section mode unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
            sectionType = sectionType.astype('int')

            sectionLoc = np.zeros(nsection, dtype='intc')
            deesse.mpds_get_array_from_int_vector(mpds_xsectionParameters.sectionLoc, 0, sectionLoc)
            sectionLoc = sectionLoc.astype('int')

    else:
        err_msg = f'{fname}: section path type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    sectionPath_parameters = DeesseXInputSectionPath(
        sectionMode=sectionMode,
        sectionPathMode=sectionPathMode,
        minSpaceX=minSpaceX,
        minSpaceY=minSpaceY,
        minSpaceZ=minSpaceZ,
        balancedFillingFlag=balancedFillingFlag,
        nsection=nsection,
        sectionType=sectionType,
        sectionLoc=sectionLoc,
        logger=logger)

    return sectionPath_parameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_section_py2C(
        section_parameters,
        sectionType,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        nv,
        logger=None):
    """
    Converts section parameters (for one section) (deesseX) from python to C.

    Parameters
    ----------
    section_parameters : :class:`DeesseXInputSection`
        section parameters (for one section) in python

    sectionType : int
        id of the section type

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

    nv : int
        number of variable(s) / attribute(s)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_xsubsiminput : \\(MPDS_XSUBSIMINPUT \\*\\)
        parameters in C
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
            try:
                im_c = img_py2C(ti, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert TI from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_IMAGE_array_setitem(mpds_xsubsiminput.trainImage, i, im_c)

    # mpds_xsubsiminput.pdfTrainImage
    if nTI > 1:
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=nTI, val=section_parameters.pdfTI,
                 logger=logger)
        try:
            mpds_xsubsiminput.pdfTrainImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
            deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
            # Raise error
            err_msg = f'{fname}: cannot convert pdfTI from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.homothetyXRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyXRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyXRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.homothetyYRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyYRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyYRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.homothetyZRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyZRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyZRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.homothetyXRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyXRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyXRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.homothetyYRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyYRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyYRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.homothetyZRatio,
                     logger=logger)
            try:
                mpds_xsubsiminput.homothetyZRatioImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert homothetyZRatio image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.rotationAzimuth,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationAzimuthImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationAzimuth image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.rotationDip,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationDipImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationDip image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=1, val=section_parameters.rotationPlunge,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationPlungeImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationPlunge image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.rotationAzimuth,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationAzimuthImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationAzimuth image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.rotationDip,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationDipImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationDip image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

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
                     nv=2, val=section_parameters.rotationPlunge,
                     logger=logger)
            try:
                mpds_xsubsiminput.rotationPlungeImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
                deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert rotationPlunge image from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

        else:
            mpds_xsubsiminput.rotationPlungeImageFlag = deesse.FALSE
            mpds_xsubsiminput.rotationPlungeValue = deesse.new_real_array(2)
            deesse.mpds_set_real_vector_from_array(mpds_xsubsiminput.rotationPlungeValue, 0,
                np.asarray(section_parameters.rotationPlunge).reshape(2))

    # mpds_xsubsiminput.searchNeighborhoodParameters
    mpds_xsubsiminput.searchNeighborhoodParameters = deesse.new_MPDS_SEARCHNEIGHBORHOODPARAMETERS_array(nv)
    for i, sn in enumerate(section_parameters.searchNeighborhoodParameters):
        try:
            sn_c = search_neighborhood_parameters_py2C(sn, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
            deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
            # Raise error
            err_msg = f'{fname}: cannot convert search neighborhood parameters from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
    try:
        mpds_xsubsiminput.simAndPathParameters = set_simAndPathParameters_C(
                section_parameters.simType,
                section_parameters.simPathType,
                section_parameters.simPathStrength,
                section_parameters.simPathPower,
                section_parameters.simPathUnilateralOrder,
                logger=logger)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
        deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
        # Raise error
        err_msg = f'{fname}: cannot set "simAndPathParameters" in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # mpds_xsubsiminput.distanceThreshold
    mpds_xsubsiminput.distanceThreshold = deesse.new_real_array(nv)
    deesse.mpds_set_real_vector_from_array(
        mpds_xsubsiminput.distanceThreshold, 0,
        np.asarray(section_parameters.distanceThreshold).reshape(nv))

    # mpds_xsubsiminput.softProbability ...
    mpds_xsubsiminput.softProbability = deesse.new_MPDS_SOFTPROBABILITY_array(nv)

    # ... for each variable ...
    for i, sp in enumerate(section_parameters.softProbability):
        try:
            sp_c = softProbability_py2C(sp,
                                        nx, ny, nz,
                                        sx, sy, sz,
                                        ox, oy, oz,
                                        logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
            deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
            # Raise error
            err_msg = f'{fname}: cannot set soft probability parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

        deesse.MPDS_SOFTPROBABILITY_array_setitem(mpds_xsubsiminput.softProbability, i, sp_c)
        # deesse.free_MPDS_SOFTPROBABILITY(sp_c)

    # mpds_xsubsiminput.maxScanFraction
    mpds_xsubsiminput.maxScanFraction = deesse.new_double_array(nTI)
    deesse.mpds_set_double_vector_from_array(
        mpds_xsubsiminput.maxScanFraction, 0,
            np.asarray(section_parameters.maxScanFraction).reshape(nTI))

    # mpds_xsubsiminput.pyramidGeneralParameters ...
    try:
        mpds_xsubsiminput.pyramidGeneralParameters = pyramidGeneralParameters_py2C(section_parameters.pyramidGeneralParameters)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
        deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
        # Raise error
        err_msg = f'{fname}: cannot set pyramid general parameters in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # mpds_xsubsiminput.pyramidParameters ...
    mpds_xsubsiminput.pyramidParameters = deesse.new_MPDS_PYRAMIDPARAMETERS_array(nv)

    # ... for each variable ...
    for i, pp in enumerate(section_parameters.pyramidParameters):
        try:
            pp_c = pyramidParameters_py2C(pp, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSubSimInput(mpds_xsubsiminput)
            deesse.free_MPDS_XSUBSIMINPUT(mpds_xsubsiminput)
            # Raise error
            err_msg = f'{fname}: cannot set pyramid parameters in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
        nv, 
        distanceType,
        logger=None):
    """
    Converts section parameters (for one section) (deesseX) from C to python.

    Parameters
    ----------
    mpds_xsubsiminput : \\(MPDS_XSUBSIMINPUT \\*\\)
        parameters in C

    sectionType : int
        id of the section type

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis

    nv : int
        number of variable(s) / attribute(s)

    distanceType : [sequence of] int(s) or str(s)
        type of distance (between pattern) for each variable

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    section_parameters : :class:`DeesseXInputSection`
        section parameters (for one section) in python
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
            TI[i] = img_C2py(im, logger=logger)

    # pdfTI
    pdfTI = None
    if nTI > 1:
        im = img_C2py(mpds_xsubsiminput.pdfTrainImage, logger=logger)
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
            im = img_C2py(mpds_xsubsiminput.homothetyXRatioImage, logger=logger)
            homothetyXRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyXRatioValue, 0, v)
            homothetyXRatio = v[0]

        homothetyYLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyYRatioImage, logger=logger)
            homothetyYRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyYRatioValue, 0, v)
            homothetyYRatio = v[0]

        homothetyZLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyZRatioImage, logger=logger)
            homothetyZRatio = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyZRatioValue, 0, v)
            homothetyZRatio = v[0]

    elif homothetyUsage == 2:
        homothetyXLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyXRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyXLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyXRatioImage, logger=logger)
            homothetyXRatio = im.val
        else:
            homothetyXRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyXRatioValue, 0, homothetyXRatio)

        homothetyYLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyYRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyYLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyYRatioImage, logger=logger)
            homothetyYRatio = im.val
        else:
            homothetyYRatio = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.homothetyYRatioValue, 0, homothetyYRatio)

        homothetyZLocal = bool(int.from_bytes(mpds_xsubsiminput.homothetyZRatioImageFlag.encode('utf-8'), byteorder='big'))
        if homothetyZLocal:
            im = img_C2py(mpds_xsubsiminput.homothetyZRatioImage, logger=logger)
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
            im = img_C2py(mpds_xsubsiminput.rotationAzimuthImage, logger=logger)
            rotationAzimuth = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationAzimuthValue, 0, v)
            rotationAzimuth = v[0]

        rotationDipLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_xsubsiminput.rotationDipImage, logger=logger)
            rotationDip = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationDipValue, 0, v)
            rotationDip = v[0]

        rotationPlungeLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_xsubsiminput.rotationPlungeImage, logger=logger)
            rotationPlunge = im.val
        else:
            v = np.zeros(1)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationPlungeValue, 0, v)
            rotationPlunge = v[0]

    elif rotationUsage == 2:
        rotationAzimuthLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationAzimuthImageFlag.encode('utf-8'), byteorder='big'))
        if rotationAzimuthLocal:
            im = img_C2py(mpds_xsubsiminput.rotationAzimuthImage, logger=logger)
            rotationAzimuth = im.val
        else:
            rotationAzimuth = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationAzimuthValue, 0, rotationAzimuth)

        rotationDipLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationDipImageFlag.encode('utf-8'), byteorder='big'))
        if rotationDipLocal:
            im = img_C2py(mpds_xsubsiminput.rotationDipImage, logger=logger)
            rotationDip = im.val
        else:
            rotationDip = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationDipValue, 0, rotationDip)

        rotationPlungeLocal = bool(int.from_bytes(mpds_xsubsiminput.rotationPlungeImageFlag.encode('utf-8'), byteorder='big'))
        if rotationPlungeLocal:
            im = img_C2py(mpds_xsubsiminput.rotationPlungeImage, logger=logger)
            rotationPlunge = im.val
        else:
            rotationPlunge = np.zeros(2)
            deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.rotationPlungeValue, 0, rotationPlunge)

    # searchNeighborhoodParameters
    searchNeighborhoodParameters = np.array(nv*[None])
    for i in range(nv):
        sn_c = deesse.MPDS_SEARCHNEIGHBORHOODPARAMETERS_array_getitem(mpds_xsubsiminput.searchNeighborhoodParameters, i)
        sn = search_neighborhood_parameters_C2py(sn_c, logger=logger)
        if sn is None:
            err_msg = f'{fname}: cannot convert search neighborhood parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
        err_msg = f'{fname}: simulation type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        err_msg = f'{fname}: simulation path type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # distanceThreshold
    distanceThreshold = np.zeros(nv, dtype=float)
    deesse.mpds_get_array_from_real_vector(mpds_xsubsiminput.distanceThreshold, 0, distanceThreshold)

    # softProbability
    softProbability = np.array(nv*[None])
    for i in range(nv):
        sp_c = deesse.MPDS_SOFTPROBABILITY_array_getitem(mpds_xsubsiminput.softProbability, i)
        sp = softProbability_C2py(sp_c, logger=logger)
        if sp is None:
            err_msg = f'{fname}: cannot convert soft probability from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        softProbability[i] = sp

    # maxScanFraction
    maxScanFraction = np.zeros(nTI, dtype='double')
    deesse.mpds_get_array_from_double_vector(mpds_xsubsiminput.maxScanFraction, 0, maxScanFraction)
    maxScanFraction = maxScanFraction.astype('float')

    # pyramidGeneralParameters
    pyramidGeneralParameters = pyramidGeneralParameters_C2py(mpds_xsubsiminput.pyramidGeneralParameters, logger=logger)
    if pyramidGeneralParameters is None:
        err_msg = f'{fname}: cannot convert pyramid general parameters from C to python'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # pyramidParameters
    pyramidParameters = np.array(nv*[None])
    for i in range(nv):
        pp_c = deesse.MPDS_PYRAMIDPARAMETERS_array_getitem(mpds_xsubsiminput.pyramidParameters, i)
        pp = pyramidParameters_C2py(pp_c, logger=logger)
        if pp is None:
            err_msg = f'{fname}: cannot convert pyramid parameters from C to python'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
        postProcessingTolerance=postProcessingTolerance,
        logger=logger)

    return section_parameters
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_py2C(deesseX_input, logger=None):
    """
    Converts deesseX input from python to C.

    Parameters
    ----------
    deesseX_input : :class:`DeesseXInput`
        deesseX input in python

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_xsiminput : \\(MPDS_XSIMINPUT \\*\\)
        deesseX input in C
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
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_msg = f'{fname}: simName is not a string'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    if len(deesseX_input.simName) >= deesse.MPDS_VARNAME_LENGTH:
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_msg = f'{fname}: simName is too long'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    deesse.mpds_x_allocate_and_set_simname(mpds_xsiminput, deesseX_input.simName)
    # mpds_xsiminput.simName = deesseX_input.simName #  works too

    # mpds_xsiminput.simImage ...
    # ... set initial image im (for simulation)
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=nv, val=deesse.MPDS_MISSING_VALUE,
             varname=deesseX_input.varname,
             logger=logger)

    # ... convert im from python to C
    try:
        mpds_xsiminput.simImage = img_py2C(im, logger=logger)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_msg = f'{fname}: cannot initialize simImage in C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

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
        deesse.mpds_x_allocate_and_set_outputReportFileName(mpds_xsiminput, deesseX_input.outputReportFileName)
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
            try:
                im_c = img_py2C(dataIm, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSimInput(mpds_xsiminput)
                deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert dataImage from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_IMAGE_array_setitem(mpds_xsiminput.dataImage, i, im_c)

    # mpds_xsiminput.ndataPointSet and mpds_xsiminput.dataPointSet
    if deesseX_input.dataPointSet is None:
        mpds_xsiminput.ndataPointSet = 0
    else:
        n = len(deesseX_input.dataPointSet)
        mpds_xsiminput.ndataPointSet = n
        mpds_xsiminput.dataPointSet = deesse.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(deesseX_input.dataPointSet):
            try:
                ps_c = ps_py2C(dataPS, logger=logger)
            except Exception as exc:
                # Free memory on C side
                deesse.MPDSFreeXSimInput(mpds_xsiminput)
                deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
                # Raise error
                err_msg = f'{fname}: cannot convert dataPointSet from python to C'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg) from exc

            deesse.MPDS_POINTSET_array_setitem(mpds_xsiminput.dataPointSet, i, ps_c)

    # mpds_xsiminput.maskImageFlag and mpds_xsiminput.maskImage
    if deesseX_input.mask is None:
        mpds_xsiminput.maskImageFlag = deesse.FALSE
    else:
        mpds_xsiminput.maskImageFlag = deesse.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=deesseX_input.mask,
                 logger=logger)
        try:
            mpds_xsiminput.maskImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: cannot convert mask from python to C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

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
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_msg = f'{fname}: normalizing type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: rescaling mode unknown'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
    try:
        mpds_xsiminput.XSectionParameters = deesseX_input_sectionPath_py2C(deesseX_input.sectionPath_parameters, logger=logger)
    except Exception as exc:
        # Free memory on C side
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_msg = f'{fname}: cannot set XSectionParameters from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

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
    #     # Free memory on C side
    #     deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #     deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
    #     # Raise error
    #     err_msg = f'{fname}: section mode unknown'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)
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
    #     # Free memory on C side
    #     deesse.MPDSFreeXSimInput(mpds_xsiminput)
    #     deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
    #     # Raise error
    #     err_msg = f'{fname}: section path type unknown'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)

    # mpds_xsiminput.XSubSimInput_<*> ...
    for sect_param in deesseX_input.section_parameters:
        if sect_param.nx != nx:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: nx in (one) section parameters invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if sect_param.ny != ny:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: ny in (one) section parameters invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if sect_param.nz != nz:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: nz in (one) section parameters invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if sect_param.nv != nv:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: nv in (one) section parameters invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        if not np.all(deesseX_input.distanceType == sect_param.distanceType):
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: distanceType (one) section parameters invalid'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

        # for d1, d2 in zip(deesseX_input.distanceType, sect_param.distanceType):
        #     if d1 != d2:
        #         # Free memory on C side
        #         deesse.MPDSFreeXSimInput(mpds_xsiminput)
        #         deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        #         err_msg = f'{fname}: distanceType (one) section parameters invalid'
        #         if logger: logger.error(err_msg)
        #         raise DeesseinterfaceError(err_msg)

        try:
            if sect_param.sectionType == 0:
                mpds_xsiminput.XSubSimInput_xy = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                            nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            elif sect_param.sectionType == 1:
                mpds_xsiminput.XSubSimInput_xz = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                            nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            elif sect_param.sectionType == 2:
                mpds_xsiminput.XSubSimInput_yz = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                            nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            elif sect_param.sectionType == 3:
                mpds_xsiminput.XSubSimInput_z = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                           nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            elif sect_param.sectionType == 4:
                mpds_xsiminput.XSubSimInput_y = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                           nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            elif sect_param.sectionType == 5:
                mpds_xsiminput.XSubSimInput_x = deesseX_input_section_py2C(sect_param, sect_param.sectionType,
                                                                           nx, ny, nz, sx, sy, sz, ox, oy, oz, nv,
                                                                            logger=logger)
            else:
                # Free memory on C side
                deesse.MPDSFreeXSimInput(mpds_xsiminput)
                deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
                # Raise error
                err_msg = f'{fname}: section type in section parameters unknown'
                if logger: logger.error(err_msg)
                raise DeesseinterfaceError(err_msg)

        except Exception as exc:
            # Free memory on C side
            deesse.MPDSFreeXSimInput(mpds_xsiminput)
            deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
            # Raise error
            err_msg = f'{fname}: cannot set XSubSimInput in C'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg) from exc

    # mpds_xsiminput.seed
    mpds_xsiminput.seed = int(deesseX_input.seed)

    # mpds_xsiminput.seedIncrement
    mpds_xsiminput.seedIncrement = int(deesseX_input.seedIncrement)

    # mpds_xsiminput.nrealization
    mpds_xsiminput.nrealization = int(deesseX_input.nrealization)

    return mpds_xsiminput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_input_C2py(mpds_xsiminput, logger=None):
    """
    Converts deesseX input from C to python.

    Parameters
    ----------
    mpds_xsiminput : \\(MPDS_XSIMINPUT \\*\\)
        deesseX input in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesseX_input : :class:`DeesseXInput`
        deesseX input in python
    """
    fname = 'deesseX_input_C2py'

    # simName
    simName = mpds_xsiminput.simName

    im = img_C2py(mpds_xsiminput.simImage, logger=logger)

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
            dataImage[i] = img_C2py(im, logger=logger)

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
        im = img_C2py(mpds_xsiminput.maskImage, logger=logger)
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
        err_msg = f'{fname}: normalizing type unknown'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
            err_msg = f'{fname}: rescaling mode unknown'
            if logger: logger.error(err_msg)
            raise DeesseinterfaceError(err_msg)

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
    sectionPath_parameters = deesseX_input_sectionPath_C2py(mpds_xsiminput.XSectionParameters, logger=logger)

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_xy, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_xz, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_yz, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_z, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_y, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        section_parameters.append(deesseX_input_section_C2py(mpds_xsiminput.XSubSimInput_x, sectionType, nx, ny, nz, nv, distanceType, logger=logger))

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
        nrealization=nrealization,
        logger=logger)

    return deesseX_input
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseX_output_C2py(mpds_xsimoutput, mpds_progressMonitor, logger=None):
    """
    Converts deesse output from C to python.

    Parameters
    ----------
    mpds_xsimoutput : \\(MPDS_XSIMOUTPUT \\*\\)
        deesseX output in C

    mpds_progressMonitor : \\(MPDS_PROGRESSMONITOR \\*\\)
        progress monitor in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesseX_output : dict
        deesseX output in python, dictionary

        `{'sim':sim,
        'sim_var_original_index':sim_var_original_index,
        'simSectionType':simSectionType,
        'simSectionStep':simSectionStep,
        'nwarning':nwarning,
        'warnings':warnings}`

        with (`nreal=mpds_xsimoutput->nreal`, the number of realization(s)):

        - sim: 1D array of :class:`geone.img.Img` of shape (nreal,)
            * `sim[i]`: i-th realisation, \
            k-th variable stored refers to the original variable \
            `sim_var_original_index[k]` \
            (get from `mpds_xsimoutput->outputSimImage[0]`)

            note: `sim=None` if `mpds_xsimoutput->outputSimImage=NULL`

        - sim_var_original_index : 1D array of ints
            * `sim_var_original_index[k]`: index of the original variable \
            (given in deesse_input) of the k-th variable stored in \
            in `sim[i]` for any i (array of length `sim[*].nv`, \
            get from `mpds_xsimoutput->originalVarIndex`)

            note: `sim_var_original_index=None`
            if `mpds_xsimoutput->originalVarIndex=NULL`

        - simSectionType : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `simSectionType[i]`: section type (id identifying which type of \
            section is used) map for the i-th realisation \
            (`mpds_xsimoutput->outputSectionTypeImage[0]`)

            note: depending on section path mode (see :class:`DeesseXInput`:
            `deesseX_input.sectionPath_parameters.sectionPathMode`) that was
            used, `simSectionType` may be of size 1 even if `nreal` is
            greater than 1, in such a case the same map is valid for all
            realizations

            note: `simSectionType=None` if
            `mpds_xsimoutput->outputSectionTypeImage=NULL`

        - simSectionStep : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `simSectionStep[i]`: section step (index of simulation by direct \
            sampling of (a bunch of) sections of same type) map for the i-th \
            realisation (`mpds_xsimoutput->outputSectionStepImage[0]`)

            note: depending on section path mode (see :class:`DeesseXInput`:
            `deesseX_input.sectionPath_parameters.sectionPathMode`) that was
            used, `simSectionStep` may be of size 1 even if `nreal` is
            greater than 1, in such a case the same map is valid for all
            realizations

            note: `simSectionStep=None` if
            `mpds_xsimoutput->outputSectionStepImage=NULL`

        nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        warnings : list of strs
            list of distinct warnings encountered (can be empty)
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
            im = img_C2py(mpds_xsimoutput.outputSimImage, logger=logger)

            nv = mpds_xsimoutput.nvarSimPerReal
            k = 0
            sim = []
            for i in range(nreal):
                sim.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                               sx=im.sx, sy=im.sy, sz=im.sz,
                               ox=im.ox, oy=im.oy, oz=im.oz,
                               nv=nv, val=im.val[k:(k+nv),...],
                               varname=im.varname[k:(k+nv)],
                               logger=logger))
                k = k + nv

            del(im)
            sim = np.asarray(sim).reshape(nreal)
            # ---

        if mpds_xsimoutput.nvarSectionType:
            # --- simSectionType ---
            im = img_C2py(mpds_xsimoutput.outputSectionTypeImage, logger=logger)

            nv = mpds_xsimoutput.nvarSectionType
            simSectionType = []
            for i in range(nv):
                simSectionType.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                          sx=im.sx, sy=im.sy, sz=im.sz,
                                          ox=im.ox, oy=im.oy, oz=im.oz,
                                          nv=1, val=im.val[i,...],
                                          varname=im.varname[i],
                                          logger=logger))

            del(im)
            simSectionType = np.asarray(simSectionType).reshape(nv)
            # ---

        if mpds_xsimoutput.nvarSectionStep:
            # --- simSectionStep ---
            im = img_C2py(mpds_xsimoutput.outputSectionStepImage, logger=logger)

            nv = mpds_xsimoutput.nvarSectionStep
            simSectionStep = []
            for i in range(nv):
                simSectionStep.append(Img(nx=im.nx, ny=im.ny, nz=im.nz,
                                          sx=im.sx, sy=im.sy, sz=im.sz,
                                          ox=im.ox, oy=im.oy, oz=im.oz,
                                          nv=1, val=im.val[i,...],
                                          varname=im.varname[i],
                                          logger=logger))

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
def deesseXRun(
        deesseX_input,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Launches deesseX.

    Parameters
    ----------
    deesseX_input : :class:`DeesseXInput`
        deesseX input in python

    nthreads : int, default: -1
        number of thread(s) to use for C program;
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
    deesseX_output : dict
        deesseX output in python, dictionary

        `{'sim':sim,
        'sim_var_original_index':sim_var_original_index,
        'simSectionType':simSectionType,
        'simSectionStep':simSectionStep,
        'nwarning':nwarning,
        'warnings':warnings}`

        with `nreal=deesseX_input.nrealization`:

        - sim: 1D array of :class:`geone.img.Img` of shape (nreal,)
            * `sim[i]`: i-th realisation, \
            k-th variable stored refers to the original variable \
            `sim_var_original_index[k]` \
            (get from `mpds_xsimoutput->outputSimImage[0]`)

            note: `sim=None` if `mpds_xsimoutput->outputSimImage=NULL`

        - sim_var_original_index : 1D array of ints
            * `sim_var_original_index[k]`: index of the original variable \
            (given in deesse_input) of the k-th variable stored in \
            in `sim[i]` for any i (array of length `sim[*].nv`, \
            get from `mpds_xsimoutput->originalVarIndex`)

            note: `sim_var_original_index=None`
            if `mpds_xsimoutput->originalVarIndex=NULL`

        - simSectionType : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `simSectionType[i]`: section type (id identifying which type of \
            section is used) map for the i-th realisation \
            (`mpds_xsimoutput->outputSectionTypeImage[0]`)

            note: depending on section path mode (see :class:`DeesseXInput`:
            `deesseX_input.sectionPath_parameters.sectionPathMode`) that was
            used, `simSectionType` may be of size 1 even if `nreal` is
            greater than 1, in such a case the same map is valid for all
            realizations

            note: `simSectionType=None` if
            `mpds_xsimoutput->outputSectionTypeImage=NULL`

        - simSectionStep : 1D array of :class:`geone.img.Img` of shape (nreal,), optional
            * `simSectionStep[i]`: section step (index of simulation by direct \
            sampling of (a bunch of) sections of same type) map for the i-th \
            realisation (`mpds_xsimoutput->outputSectionStepImage[0]`)

            note: depending on section path mode (see :class:`DeesseXInput`:
            `deesseX_input.sectionPath_parameters.sectionPathMode`) that was
            used, `simSectionStep` may be of size 1 even if `nreal` is
            greater than 1, in such a case the same map is valid for all
            realizations

            note: `simSectionStep=None` if
            `mpds_xsimoutput->outputSectionStepImage=NULL`

        nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'deesseXRun'

    if not deesseX_input.ok:
        err_msg = f'{fname}: check deesseX input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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

    if verbose > 1:
        if logger:
            logger.info(f"{fname}: DeeSseX running... [" + \
                f"VERSION {deesse.MPDS_X_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_X_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
        else:
            print(f"{fname}: DeeSseX running... [" + \
                f"VERSION {deesse.MPDS_X_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_X_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching deesseX...

    # Convert deesseX input from python to C
    try:
        mpds_xsiminput = deesseX_input_py2C(deesseX_input, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert deesseX input from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # if mpds_xsiminput is None:
    #     err_msg = f'{fname}: cannot convert deesseX input from python to C'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)

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
    deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)

    if err:
        # Free memory on C side: simulation output
        deesse.MPDSFreeXSimOutput(mpds_xsimoutput)
        deesse.free_MPDS_XSIMOUTPUT(mpds_xsimoutput)
        # Free memory on C side: progress monitor
        deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    deesseX_output = deesseX_output_C2py(mpds_xsimoutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: simulation output
    deesse.MPDSFreeXSimOutput(mpds_xsimoutput)
    deesse.free_MPDS_XSIMOUTPUT(mpds_xsimoutput)

    # Free memory on C side: progress monitor
    deesse.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose > 1 and deesseX_output:
        if logger:
            logger.info(f'{fname}: DeeSseX run complete')
        else:
            print(f'{fname}: DeeSseX run complete')

    # Show (print) encountered warnings
    if verbose > 0 and deesseX_output and deesseX_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({deesseX_output['nwarning']} times in all):")
            for i, warning_message in enumerate(deesseX_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return deesseX_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def deesseXRun_mp(
        deesseX_input,
        nproc=-1,
        nthreads_per_proc=-1,
        verbose=2,
        logger=None):
    """
    Computes the same as the function :func:`deesseXRun`, using multiprocessing.

    All the parameters are the same as those of the function :func:`deesseXRun`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`deesseXRun`];
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`deesseXRun`.

    **Parameters (new)**
    --------------------
    nproc : int, default: -1
        number of process(es): a negative number (or zero), -n <= 0, can be specified 
        to use the total number of cpu(s) of the system except n; `nproc` is finally
        at maximum equal to `nreal` but at least 1 by applying:
        
        - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
        - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
        where nmax is the total number of cpu(s) of the system (retrieved by \
        `multiprocessing.cpu_count()`)

        note: if `nproc=None`, `nproc=-1` is used

    nthreads_per_proc : int, default: -1
        number of thread(s) per process;
        if `nthreads_per_proc = -n <= 0`: `nthreads_per_proc` is automatically 
        computed as the maximal integer (but at least 1) such that 
        `nproc*nthreads_per_proc <= nmax-n`, where nmax is the total number of cpu(s)
        of the system (retrieved by `multiprocessing.cpu_count()`); 

        note: if `nthreads_per_proc=None`, `nthreads_per_proc=-1` is used
    """
    fname = 'deesseXRun_mp'

    if not deesseX_input.ok:
        err_msg = f'{fname}: check deesseX input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # if deesseX_input.nrealization <= 1:
    #     if verbose > 1:
    #         if logger:
    #             logger.info(f'{fname}: number of realization does not exceed 1: launching deesseXRun...')
    #         else:
    #             print(f'{fname}: number of realization does not exceed 1: launching deesseXRun...')
    #     nthreads = nthreads_per_proc
    #     if nthreads is None:
    #         nthreads = -1
    #     deesseX_output = deesseXRun(deesseX_input, nthreads=nthreads, verbose=verbose, logger=logger)
    #     return deesseX_output

    # Set number of process(es): nproc
    if nproc is None:
        nproc = -1
    
    if nproc <= 0:
        nproc = max(min(multiprocessing.cpu_count() + nproc, deesseX_input.nrealization), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), deesseX_input.nrealization), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of processes has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of processes has been changed (now: nproc={nproc})')

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
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(deesseX_input.nrealization, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info(f"{fname}: DeeSseX running on {nproc} process(es)... [" + \
                f"VERSION {deesse.MPDS_X_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_X_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
        else:
            print(f"{fname}: DeeSseX running on {nproc} process(es)... [" + \
                f"VERSION {deesse.MPDS_X_VERSION_NUMBER:s} / " + \
                f"BUILD NUMBER {deesse.MPDS_X_BUILD_NUMBER:s} / " + \
                f"OpenMP {nth:d} thread(s)]")
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
        verb = 0
        # if i==0:
        #     verb = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verb = 0
        # Launch deesseX (i-th process)
        out_pool.append(pool.apply_async(deesseXRun, args=(input, nth, verb), kwds={'logger':logger}))

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
    nwarning = int(np.sum([out['nwarning'] for out in deesseX_output_proc]))
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
            for i in range(deesseX_input.nrealization):
                for k in range(simSectionType[i].nv):
                    simSectionType[i].varname[k] = simSectionType[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
        else: # keep only first map (all are the same)
            simSectionType = np.array([simSectionType[0]])
    if simSectionStep is not None:
        if deesseX_input.sectionPath_parameters.sectionPathMode == 'section_path_random':
            for i in range(deesseX_input.nrealization):
                for k in range(simSectionStep[i].nv):
                    simSectionStep[i].varname[k] = simSectionStep[i].varname[k][:-ndigit] + f'{i:0{ndigit}d}'
        else: # keep only first map (all are the same)
            simSectionStep = np.array([simSectionStep[0]])

    deesseX_output = {
        'sim':sim, 'sim_var_original_index':sim_var_original_index,
        'simSectionType':simSectionType, 'simSectionStep':simSectionStep,
        'nwarning':nwarning, 'warnings':warnings
        }

    if verbose > 1 and deesseX_output:
        if logger:
            logger.info(f'{fname}: DeeSseX run complete (all process(es))')
        else:
            print(f'{fname}: DeeSseX run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and deesseX_output and deesseX_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({deesseX_output['nwarning']} times in all):")
            for i, warning_message in enumerate(deesseX_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return deesseX_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def exportDeesseXInput(
        deesseX_input,
        dirname='input_ascii',
        fileprefix='dsX',
        endofline='\n',
        verbose=1,
        logger=None):
    """
    Exports deesseX input in txt (ASCII) files (in the directory `dirname`).

    The command line version of deesseX can then be launched from the directory
    `dirname` by using the generated txt files.

    Parameters
    ----------
    deesseX_input : :class:`DeesseXInput`
        deesse input in python

    dirname : str, default: 'input_ascii'
        name of the directory in which the files will be written;
        if not existing, it will be created;
        WARNING: the generated files might erase already existing ones!

    fileprefix : str, default: 'dsX'
        prefix for generated files, the main input file will be
        "`dirname`/`fileprefix`.in"

    endofline : str, default: '\\\\n'
        end of line character

    verbose : int, default: 1
        verbose mode for comments in the written main input file:

        - 0: no comment
        - 1: basic comments
        - 2: detailed comments

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    fname = 'exportDeesseXInput'

    if not deesseX_input.ok:
        err_msg = f'{fname}: check deesseX input'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Create ouptut directory if needed
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Convert deesseX input from python to C
    try:
        mpds_xsiminput = deesseX_input_py2C(deesseX_input, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert deesseX input from python to C'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg) from exc

    # if mpds_xsiminput is None:
    #     err_msg = f'{fname}: cannot convert deesseX input from python to C'
    #     if logger: logger.error(err_msg)
    #     raise DeesseinterfaceError(err_msg)

    err = deesse.MPDSExportXSimInput( mpds_xsiminput, dirname, fileprefix, endofline, verbose)

    if err:
        # Free memory on C side: deesseX input
        deesse.MPDSFreeXSimInput(mpds_xsiminput)
        deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
        # Raise error
        err_message = deesse.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Free memory on C side: deesseX input
    deesse.MPDSFreeXSimInput(mpds_xsiminput)
    deesse.free_MPDS_XSIMINPUT(mpds_xsiminput)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def importDeesseXInput(filename, dirname='.', logger=None):
    """
    Imports deesseX input from txt (ASCII) files.

    The files used for command line version of deesseX (from the directory named
    `dirname`) are read and the corresponding deesseX input in python is
    returned.

    Parameters
    ----------
    filename : str
        name of the main input txt (ASCII) file (without path) used for the
        command line version of deesseX

    dirname : str, default: '.'
        name of the directory in which the main input txt (ASCII) file is stored
        (and from which the command line version of deesseX would be launched)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    deesseX_input :class:`DeesseXInput`
        deesseX input in python
    """
    fname = 'importDeesseXInput'

    # Check directory
    if not os.path.isdir(dirname):
        err_msg = f'{fname}: directory does not exist'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    # Check file
    if not os.path.isfile(os.path.join(dirname, filename)):
        err_msg = f'{fname}: input file does not exist'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

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
        deesseX_input = deesseX_input_C2py(mpds_xsiminput, logger=logger)

    except:
        # Free memory on C side: deesseX input
        deesse.delete_MPDS_XSIMINPUTp(mpds_xsiminputp)
        # Raise error
        err_msg = f'{fname}: cannot import deesseX input from ASCII files'
        if logger: logger.error(err_msg)
        raise DeesseinterfaceError(err_msg)

    finally:
        # Change directory (to initial working directory)
        os.chdir(cwd)

    # Free memory on C side: deesseX input
    deesse.delete_MPDS_XSIMINPUTp(mpds_xsiminputp)

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

            - sim : (1-dimensional array of Img (class) of size nreal)
                * sim[i]: i-th realisation

            - path : (1-dimensional array of Img (class) of size nreal or None)
                * path[i]: path index map for the i-th realisation \
                (path is `None` if `deesse_input.outputPathIndexFlag=False`)

            - error:  (1-dimensional array of Img (class) of size nreal or `None`)
                * error[i]: error map for the i-th realisation \
                (path is `None` if `deesse_input.outputErrorFlag=False`)
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
