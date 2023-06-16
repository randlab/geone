#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'blockdata.py'
author:         Julien Straubhaar
date:           feb-2018

Definition of class for block data, and relative functions.
"""

import numpy as np
import os

# ============================================================================
class BlockData(object):
    """
    Defines block data (for one variable):
        blockDataUsage:
                    (int) indicates the usage of block data:
                        - 0: no block data
                        - 1: block data: block mean value

        nblock:     (int) number of block(s) (unused if blockDataUsage == 0)

        nodeIndex:  (list of nblock 2-dimensional array of ints with 3 columns)
                        node index in each block, nodeIndex[i] is a (n_i, 3)
                        array a containing the node index in the simulation grid
                        along each coordinate for the i-th block, n_i beeing
                        the number of nodes in that block
                        (unused if blockDataUsage == 0)

        value:      (1-dimensional array of floats of size nblock)
                        target value for each block
                        (unused if blockDataUsage == 0)

        tolerance:  (1-dimensional array of floats of size nblock)
                        tolerance for each block
                        (unused if blockDataUsage == 0)

        activatePropMin:
                    (1-dimensional array of floats of size nblock)
                        minimal proportion of informed nodes in the block,
                        under which the block data constraint is deactivated,
                        for each block
                        (unused if blockDataUsage == 0)

        activatePropMax:
                    (1-dimensional array of floats of size nblock)
                        maximal proportion of informed nodes in the block,
                        above which the block data constraint is deactivated,
                        for each block
                        (unused if blockDataUsage == 0)
    """

    def __init__(self,
                 blockDataUsage=0,
                 nblock=0,
                 nodeIndex=None,
                 value=None,
                 tolerance=None,
                 activatePropMin=None,
                 activatePropMax=None):
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
def readBlockData(filename):
    """
    Reads block data from a file (ASCII):

    :param filename:        (string) name of the file

    :return:                (BlockData class) block data
    """

    fname = 'readBlockData'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

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
    Writes block data in a file (ASCII):

    :param filename:    (string) name of the file
    :param fmt:         (string) format for value, toleance and activate
                            proportions
    """

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
