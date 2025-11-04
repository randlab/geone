#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'blockdata.py'
# author:         Julien Straubhaar
# date:           feb-2018
# -------------------------------------------------------------------------

"""
Module for managing block data (for deesse), and relative functions.
"""

import numpy as np
import os

# ============================================================================
class BlockDataError(Exception):
    """
    Custom exception related to `blockdata` module.
    """
    pass
# ============================================================================

# ============================================================================
class BlockData(object):
    """
    Class defining block data for one variable (for deesse).

    **Attributes**

    blockDataUsage : int, default: 0
        defines the usage of block data:

        - `blockDataUsage=0`: no block data
        - `blockDataUsage=1`: block data defined as block mean value

    nblock : int, default: 0
        number of block(s) (used if `blockDataUsage=1`)

    nodeIndex : sequence of 2D array-like of ints with 3 columns, optional
        node index in each block (used if `blockDataUsage=1`):

        - `nodeIndex[i][j]`: sequence of 3 floats, \
        node index in the simulation grid along x, y, z axis \
        of the j-th node of the i-th block

    value : sequence of floats of length `nblock`, optional
        target value for each block (used if `blockDataUsage=1`)

    tolerance : sequence of floats of length `nblock`, optional
        tolerance for each block (used if `blockDataUsage=1`)

    activatePropMin : sequence of floats of length `nblock`, optional
        minimal proportion of informed nodes in the block, under which the block
        data constraint is deactivated, for each block (used if `blockDataUsage=1`)

    activatePropMax : sequence of floats of length `nblock`, optional
        maximal proportion of informed nodes in the block, above which the block
        data constraint is deactivated, for each block (used if `blockDataUsage=1`)

    **Methods**
    """
    def __init__(self,
                 blockDataUsage=0,
                 nblock=0,
                 nodeIndex=None,
                 value=None,
                 tolerance=None,
                 activatePropMin=None,
                 activatePropMax=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        blockDataUsage : int, default: 0
            defines the usage of block data

        nblock : int, default: 0
            number of block(s)

        nodeIndex : sequence of 2D array-like of ints with 3 columns, optional
            node index in each block

        value : sequence of floats of length nblock, optional
            target value for each block

        tolerance : sequence of floats of length nblock, optional
            tolerance for each block

        activatePropMin : sequence of floats of length nblock, optional
            minimal proportion of informed nodes in the block, under which the block
            data constraint is deactivated, for each block

        activatePropMax : sequence of floats of length nblock, optional
            maximal proportion of informed nodes in the block, above which the block
            data constraint is deactivated, for each block
        """
        # fname = 'BlockData'

        self.blockDataUsage = blockDataUsage
        self.nblock = nblock
        self.nodeIndex = nodeIndex

        if nodeIndex is None:
            self.nodeIndex = None
        else:
            self.nodeIndex = [np.asarray(x) for x in self.nodeIndex]

        if value is None:
            self.value = None
        else:
            self.value = np.asarray(value, dtype=float).reshape(nblock)

        if tolerance is None:
            self.tolerance = None
        else:
            self.tolerance = np.asarray(tolerance, dtype=float).reshape(nblock)

        if activatePropMin is None:
            self.activatePropMin = None
        else:
            self.activatePropMin = np.asarray(activatePropMin, dtype=float).reshape(nblock)

        if activatePropMax is None:
            self.activatePropMax = None
        else:
            self.activatePropMax = np.asarray(activatePropMax, dtype=float).reshape(nblock)

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** BlockData object ***'
        out = out + '\n' + 'blockDataUsage = {0.blockDataUsage}:'.format(self)
        if self.blockDataUsage == 0:
            out = out + ' no block data'
        elif self.blockDataUsage == 1:
            out = out + ' block mean value'
        else:
            out = out + ' unknown'
        if self.blockDataUsage == 1:
            out = out + '\n' + 'nblock = {0.nblock} # number of blocks'.format(self)
            out = out + '\n' + '    parameters for each block in fields'
            out = out + '\n' + '    ".nodeIndex", ".value", ".tolerance", ".activatePropMin", ".activatePropMax"'
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------
# ============================================================================

# ----------------------------------------------------------------------------
def readBlockData(filename, logger=None):
    """
    Reads block data from a txt file.

    Parameters
    ----------
    filename : str
        name of the file

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    bd : :class:`BlockData`
        block data
    """
    fname = 'readBlockData'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise BlockDataError(err_msg)

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read number of block (1st line)
        nblock = int(ff.readline())

        # Initialize fields ...
        nodeIndex = []
        value = np.zeros(nblock)
        tolerance = np.zeros(nblock)
        activatePropMin = np.zeros(nblock)
        activatePropMax = np.zeros(nblock)

        # Read "blocks"...
        for i in range(nblock):
            li = ff.readline()
            t = [x for x in li.split()]
            nnode = int(t[0])
            value[i], tolerance[i], activatePropMin[i], activatePropMax[i] = [float(x) for x in t[1:5]]
            nodeIndex.append(np.array([[int(j) for j in ff.readline().split()] for k in range(nnode)]))

    # Set block data
    bd = BlockData(blockDataUsage=1,
                   nblock=nblock,
                   nodeIndex=nodeIndex,
                   value=value,
                   tolerance=tolerance,
                   activatePropMin=activatePropMin,
                   activatePropMax=activatePropMax)

    return bd
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeBlockData(bd, filename, fmt='.5g'):
    """
    Writes block data in a txt file.

    Parameters
    ----------
    bd : :class:`BlockData`
        block data

    filename : str
        name of the file

    fmt : str, default: '.5g'
        format string for target value, tolerance and active proportion (min
        and max) in the block data
    """
    # fname = 'writeBlockData'

    if bd.blockDataUsage == 0:
        return None

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        # Write the number of block(s)
        ff.write('{}\n'.format(bd.nblock).encode())

        # Write "blocks"...
        for ni, v, t, amin, amax in zip(bd.nodeIndex, bd.value,
                                        bd.tolerance, bd.activatePropMin,
                                        bd.activatePropMax):
            ff.write('{} {:{fmt}} {:{fmt}} {:{fmt}} {:{fmt}}\n'.format(len(ni), v, t, amin, amax, fmt=fmt).encode())
            np.savetxt(ff, np.asarray(ni), delimiter=' ', fmt="%g")
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.blockdata'.")
