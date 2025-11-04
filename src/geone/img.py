#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'img.py'
# author:         Julien Straubhaar
# date:           jan-2018
# -------------------------------------------------------------------------

"""
Module for "images" and "point sets", and relative functions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy

# ============================================================================
class ImgError(Exception):
    """
    Custom exception related to `img` module.
    """
    pass
# ============================================================================

# ============================================================================
class Img(object):
    """
    Class defining an image as a regular 3D-grid with variables attached to cells.

    **Attributes**

    nx : int, default: 1
        number of grid cells along x axis

    ny : int, default: 1
        number of grid cells along y axis

    nz : int, default: 1
        number of grid cells along z axis

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    sx : float, default: 1.0
        cell size along x axis

    sy : float, default: 1.0
        cell size along y axis

    sz : float, default: 1.0
        cell size along z axis

        Note: `(sx, sy, sz)` is the cell size

    ox : float, default: 0.0
        origin of the grid along x axis (x coordinate of cell border)

    oy : float, default: 0.0
        origin of the grid along y axis (y coordinate of cell border)

    oz : float, default: 0.0
        origin of the grid along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    nv : int, default: 0
        number of variable(s) / attribute(s)

    val : 4D array of float of shape (`nv`, `nz`, `ny`, `nx`)
        attribute(s) / variable(s) values:

        - `val[iv, iz, iy, ix]`: value of the variable iv attached to the \
        grid cell of index `iz`, `iy`, `ix` along axis z, y, x respectively

    varname : list of str, of length `nv`
        variable names:

        - `varname[iv]`: name of the variable `iv`

    name : str
        name of the image

    **Methods**
    """
    #
    # Methods
    # -------
    # set_default_varname()
    #     Sets default variable names for each variable.
    # set_varname(varname=None, ind=-1)
    #     Sets name for a given variable.
    # set_dimension(nx, ny, nz, newval=np.nan)
    #     Sets dimension of the grid, i.e. number of cells along each axis.
    # set_spacing(sx, sy, sz)
    #     Sets spacing, i.e. cell size along each axis.
    # set_origin(ox, oy, oz)
    #     Sets grid origin (bottom-lower-left corner).
    # set_grid(nx, ny, nz, sx, sy, sz, ox, oy, oz, newval=np.nan)
    #     Sets grid geometry (dimension, cell size, and origin).
    # resize()
    #     Resizes the image (including "variable" direction).
    # insert_var(val=np.nan, varname=None, ind=0)
    #     Inserts one or several variable(s) at a given index.
    # append_var(val=np.nan, varname=None)
    #     Appends (i.e. inserts at the end) one or several variable(s).
    # remove_var(ind=None, indList=None)
    #     Removes variable(s) of given index(es).
    # remove_allvar()
    #     Removes all variables.
    # set_var(val=np.nan, varname=None, ind=-1)
    #     Sets values and name of one variable (of given index).
    # extract_var(ind=None, indList=None)
    #     Extracts variable(s) (of given index(es)).
    # get_unique_one_var(ind=0, ignore_missing_value=True)
    #     Gets unique values of one variable (of given index).
    # get_prop_one_var(ind=0, density=True, ignore_missing_value=True)
    #     Gets proportions (density or count) of unique values of one variable (of given index).
    # get_unique(ignore_missing_value=True)
    #     Gets unique values over all the variables
    # get_prop(density=True, ignore_missing_value=True)
    #     Gets proportions (density or count) of unique values for each variable.
    # flipx()
    #     Flips variable values according to x axis.
    # flipy()
    #     Flips variable values according to y axis.
    # flipz()
    #     Flips variable values according to z axis.
    # flipv()
    #     Flips variable values according to v (variable) axis.
    # permxy()
    #     Permutes / swaps x and y axes.
    # permxz()
    #     Permutes / swaps x and z axes.
    # permyz()
    #     Permutes / swaps y and z axes.
    # swap_xy()
    #     Swaps x and y axes.
    # swap_xz()
    #     Swaps x and z axes.
    # swap_yz()
    #     Swaps y and z axes.
    # transpose_xyz_to_xzy()
    #     Applies transposition: sends original x, y, z axes to x, z, y axes.
    # transpose_xyz_to_yxz()
    #     Applies transposition: sends original x, y, z axes to y, x, z axes.
    # transpose_xyz_to_yzx()
    #     Applies transposition: sends original x, y, z axes to y, z, x axes.
    # transpose_xyz_to_zxy()
    #     Applies transposition: sends original x, y, z axes to z, x, y axes.
    # transpose_xyz_to_zyx()
    #     Applies transposition: sends original x, y, z axes to z, y, x axes.
    # nxyzv()
    #     Returns the size of the array `val`, i.e. number of variables times number of grid cells.
    # nxyz()
    #     Returns the number of grid cells.
    # nxy()
    #     Returns the number of grid cells in a xy-section.
    # nxz()
    #     Returns the number of grid cells in a xz-section.
    # nyz()
    #     Returns the number of grid cells in a yz-section.
    # xmin()
    #     Returns min coordinate of the grid along x axis.
    # ymin()
    #     Returns min coordinate of the grid along y axis.
    # zmin()
    #     Returns min coordinate of the grid along z axis.
    # xmax()
    #     Returns max coordinate of the grid along x axis.
    # ymax()
    #     Returns max coordinate of the grid along y axis.
    # zmax()
    #     Returns max coordinate of the grid along z axis.
    # x()
    #     Returns 1D array of "unique" x coordinates of the grid cell centers.
    # y()
    #     Returns 1D array of "unique" y coordinates of the grid cell centers.
    # z()
    #     Returns 1D array of "unique" z coordinates of the grid cell centers.
    # xx()
    #     Returns 3D array of x coordinates of the grid cell centers.
    # yy()
    #     Returns 3D array of y coordinates of the grid cell centers.
    # zz()
    #     Returns 3D array of z coordinates of the grid cell centers.
    # ix()
    #     Returns 1D array of "unique" index of grid cell along x axis.
    # iy()
    #     Returns 1D array of "unique" index of grid cell along y axis.
    # iz()
    #     Returns 1D array of "unique" index of grid cell along z axis.
    # ixx()
    #     Returns 3D array of index along x axis of the grid cells.
    # iyy()
    #     Returns 3D array of index along y axis of the grid cells.
    # izz()
    #     Returns 3D array of index along z axis of the grid cells.
    # vmin()
    #     Returns 1D array of min value of each variable, ignoring `numpy.nan` entries.
    # vmax()
    #     Returns 1D array of max value of each variable, ignoring `numpy.nan` entries.
    #
    def __init__(self,
                 nx=1,   ny=1,   nz=1,
                 sx=1.0, sy=1.0, sz=1.0,
                 ox=0.0, oy=0.0, oz=0.0,
                 nv=0, val=np.nan, varname=None,
                 name='',
                 logger=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        nx : int, default: 1
            number of grid cells along x axis

        ny : int, default: 1
            number of grid cells along y axis

        nz : int, default: 1
            number of grid cells along z axis

            Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

        sx : float, default: 1.0
            cell size along x axis

        sy : float, default: 1.0
            cell size along y axis

        sz : float, default: 1.0
            cell size along z axis

            Note: `(sx, sy, sz)` is the cell size

        ox : float, default: 0.0
            origin of the grid along x axis (x coordinate of cell border)

        oy : float, default: 0.0
            origin of the grid along y axis (y coordinate of cell border)

        oz : float, default: 0.0
            origin of the grid along z axis (z coordinate of cell border)

            Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

        nv : int, default: 0
            number of variable(s) / attribute(s) attached to the grid cells

        val : float or array-like of size `nv*nz*ny*nx`
            attribute(s) / variable(s) values

        varname : str or 1D array-like of strs of length `nv`, optional
            variable name(s); if one variable name is given for multiple
            variables, the variable index is used as suffix;
            by default (`None`): variable names are set to "V<num>",
            where <num> starts from 0

        name : str, default: ''
            name of the image

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'Img'

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

        valarr = np.asarray(val, dtype=float) # possibly 0-dimensional
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(nx*ny*nz*nv)
        elif valarr.size != nx*ny*nz*nv:
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        self.val = valarr.reshape(nv, nz, ny, nx)

        if varname is None:
            self.varname = [f'V{i:d}' for i in range(nv)]
        else:
            varname = list(np.asarray(varname).reshape(-1))
            if len(varname) == nv:
                self.varname = varname
            elif len(varname) == 1: # more than one variable and only one varname
                self.varname = [f'{varname[0]}{i:d}' for i in range(nv)]
            else:
                err_msg = f'{fname}: `varname` has not an acceptable size'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        self.name = name

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** Img object ***'
        out = out + '\n' + "name = '{0.name}'".format(self)
        out = out + '\n' + '(nx, ny, nz) = ({0.nx}, {0.ny}, {0.nz}) # number of cells along each axis'.format(self)
        out = out + '\n' + '(sx, sy, sz) = ({0.sx}, {0.sy}, {0.sz}) # cell size (spacing) along each axis'.format(self)
        out = out + '\n' + '(ox, oy, oz) = ({0.ox}, {0.oy}, {0.oz}) # origin (coordinates of bottom-lower-left corner)'.format(self)
        out = out + '\n' + 'nv = {0.nv}  # number of variable(s)'.format(self)
        out = out + '\n' + 'varname = {0.varname}'.format(self)
        out = out + '\n' + 'val: {0.val.shape}-array'.format(self)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_default_varname(self):
        """
        Sets default variable names: varname = ('V0', 'V1', ...).
        """
        # fname = 'set_default_varname'

        self.varname = [f'V{i:d}' for i in range(self.nv)]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_varname(self, varname=None, ind=-1, logger=None):
        """
        Sets name of the variable of the given index.

        Parameters
        ----------
        varname : str, optional
            name to be set; by default (`None`): "V" followed by the variable
            index is used

        ind : int, default: -1
            index of the variable for which the name is given (negative integer
            for indexing from the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'set_varname'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if varname is None:
            varname = f'V{ii:d}'

        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_dimension(self, nx, ny, nz, newval=np.nan):
        """
        Sets dimension of the grid, i.e. number of cells along each axis.

        Sets grid dimension and updates the array of variables values (by
        truncation or extension if needed).

        Parameters
        ----------
        nx : int
            number of grid cells along x axis

        ny : int
            number of grid cells along y axis

        nz : int
            number of grid cells along z axis

        newval : float, default: `numpy.nan`
            new value to be inserted in the array `.val` (if needed)
        """
        # fname = 'set_dimension'

        # Truncate val array (along reduced dimensions)
        self.val = self.val[:, 0:nz, 0:ny, 0:nx]

        # Extend val array when needed:
        for (n, i) in zip([nx, ny, nz], [3, 2, 1]):
            if n > self.val.shape[i]:
                s = [j for j in self.val.shape]
                s[i] = n - self.val.shape[i]
                self.val = np.concatenate((self.val, newval * np.ones(s)), i)

        # Update nx, ny, nz
        self.nx = nx
        self.ny = ny
        self.nz = nz
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_spacing(self, sx=1.0, sy=1.0, sz=1.0):
        """
        Sets spacing, i.e. cell size along each axis.

        Parameters
        ----------
        sx : float, default: 1.0
            cell size along x axis

        sy : float, default: 1.0
            cell size along y axis

        sz : float, default: 1.0
            cell size along z axis
        """
        # fname = 'set_spacing'

        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_origin(self, ox=0.0, oy=0.0, oz=0.0):
        """
        Sets grid origin (bottom-lower-left corner).

        Parameters
        ----------
        ox : float, default: 0.0
            origin of the grid along x axis (x coordinate of cell border)

        oy : float, default: 0.0
            origin of the grid along y axis (y coordinate of cell border)

        oz : float, default: 0.0
            origin of the grid along z axis (z coordinate of cell border)
        """
        # fname = 'set_origin'

        self.ox = float(ox)
        self.oy = float(oy)
        self.oz = float(oz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_grid(
            self,
            nx=1,   ny=1,   nz=1,
            sx=1.0, sy=1.0, sz=1.0,
            ox=0.0, oy=0.0, oz=0.0,
            newval=np.nan):
        """
        Sets grid geometry (dimension, cell size, and origin).

        Parameters
        ----------
        nx : int, default: 1
            number of grid cells along x axis

        ny : int, default: 1
            number of grid cells along y axis

        nz : int, default: 1
            number of grid cells along z axis

            Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

        sx : float, default: 1.0
            cell size along x axis

        sy : float, default: 1.0
            cell size along y axis

        sz : float, default: 1.0
            cell size along z axis

            Note: `(sx, sy, sz)` is the cell size

        ox : float, default: 0.0
            origin of the grid along x axis (x coordinate of cell border)

        oy : float, default: 0.0
            origin of the grid along y axis (y coordinate of cell border)

        oz : float, default: 0.0
            origin of the grid along z axis (z coordinate of cell border)

            Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

        newval : float, default: `numpy.nan`
            new value to be inserted in the array `.val` (if needed)
        """
        # fname = 'set_grid'

        self.set_dimension(nx, ny, nz, newval)
        self.set_spacing(sx, sy, sz)
        self.set_origin(ox, oy, oz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def resize(
            self,
            ix0=0, ix1=None,
            iy0=0, iy1=None,
            iz0=0, iz1=None,
            iv0=0, iv1=None,
            newval=np.nan, 
            newvarname="",
            logger=None):
        """
        Resizes the image (including "variable" direction).

        Keeps slices from `ix0` (included) to `ix1` (excluded) (with step of 1)
        along x direction, and similarly for y, z and v (variable) direction.
        The origin (`.ox`, `.oy`, `.oz`) is updated accordingly. New value `newval`
        is inserted at possible new locations and new variable name `newvarname`
        (followed by an index) is used for possible new variable(s).

        Parameters
        ----------
        ix0 : int, default: 0
            index of first slice along x direction

        ix1 : int, optional
            1+index of last slice along x direction (`ix0` < `ix1`);
            by default (`None`): number of cells in x direction (`.nx`) is used

        iy0 : int, default: 0
            index of first slice along y direction

        iy1 : int, optional
            1+index of last slice along y direction (`iy0` < `iy1`);
            by default (`None`): number of cells in y direction (`.ny`) is used

        iz0 : int, default: 0
            index of first slice along z direction

        iz1 : int, optional
            1+index of last slice along z direction (`iz0` < `iz1`);
            by default (`None`): number of cells in z direction (`.nz`) is used

        iv0 : int, default: 0
            index of first slice along variable indices

        iv1 : int, optional
            1+index of last slice along variables indices (`iv0` < `iv1`);
            by default (`None`): number of variables (`.nv`) is used

        newval : float, default: `numpy.nan`
            new value to be inserted in the array `.val` (if needed)

        newvarname : str, default: ''
            prefix for new variable (if needed)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'resize'

        if ix1 is None:
            ix1 = self.nx

        if iy1 is None:
            iy1 = self.ny

        if iz1 is None:
            iz1 = self.nz

        if iv1 is None:
            iv1 = self.nv

        if ix0 >= ix1:
            err_msg = f'{fname}: invalid index(es) along x'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if iy0 >= iy1:
            err_msg = f'{fname}: invalid index(es) along y'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if iz0 >= iz1:
            err_msg = f'{fname}: invalid index(es) along z'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if iv0 >= iv1:
            err_msg = f'{fname}: invalid index(es) along v'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        initShape = self.val.shape

        # Compute number of cell(s) to prepend (n0) and to append (n1) in each
        # direction
        n0 = -np.minimum([iv0, iz0, iy0, ix0], 0) # element-wise minimum
        n1 = np.maximum(np.array([iv1, iz1, iy1, ix1]) - self.val.shape, 0) # element-wise minimum

        # Truncate val array (along reduced dimensions)
        self.val = self.val[np.max([iv0, 0]):iv1,
                            np.max([iz0, 0]):iz1,
                            np.max([iy0, 0]):iy1,
                            np.max([ix0, 0]):ix1]

        # Extend val array when needed:
        for i in range(4):
            s0 = [j for j in self.val.shape]
            s1 = [j for j in self.val.shape]
            s0[i] = n0[i]
            s1[i] = n1[i]
            self.val = np.concatenate((newval * np.ones(s0), self.val, newval * np.ones(s1)), i)

        # Update varname
        self.varname = [f'{newvarname}{i}' for i in range(n0[0])] + \
                       [self.varname[i] for i in range(np.max([iv0, 0]), np.min([iv1, initShape[0]]))] + \
                       [f'{newvarname}{n0[0]+i}' for i in range(n1[0])]

        # Update nx, ny, nz, nv
        self.nv, self.nz, self.ny, self.nx = self.val.shape

        # Update ox, oy, oz
        self.ox = self.ox + ix0 * self.sx
        self.oy = self.oy + iy0 * self.sy
        self.oz = self.oz + iz0 * self.sz
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, val=np.nan, varname=None, ind=0, logger=None):
        """
        Inserts one or several variable(s) at a given index.

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of the new variable(s); the size of the array must be

            - a multiple of the number of grid cells (i.e. `.nx * .ny * .nz`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all grid cells

        varname : str or 1D array-like of strs, optional
            name(s) of the new variable(s);
            by default (`None`): variable names are set to "V<num>", where <num>
            starts from the number of variables before the insertion

        ind : int, default: 0
            index where the new variable(s) is (are) inserted (negative integer
            for indexing from the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'insert_var'

        # Check / set ind
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Check val, set valarr (array of values)
        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size % self.nxyz() != 0:
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        m = valarr.size // self.nxyz() # number of variable(s) to be inserted

        # Check / set varname
        if varname is not None:
            if isinstance(varname, str):
                varname = [varname]
            elif (not isinstance(varname, tuple) and not isinstance(varname, list) and not isinstance(varname, np.ndarray)) or len(varname)!=m:
                err_msg = f'{fname}: `varname` does not have an acceptable size'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)
            else:
                varname = list(varname)
        else:
            # set default varname
            varname = [f'V{i:d}' for i in range(self.nv, self.nv+m)]

        # Extend val
        self.val = np.concatenate((self.val[0:ii,...],
                                  valarr.reshape(-1, self.nz, self.ny, self.nx),
                                  self.val[ii:,...]),
                                  0)
        # Extend varname list
        self.varname = self.varname[:ii] + varname + self.varname[ii:]

        # Update nv
        self.nv = self.nv + m
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def append_var(self, val=np.nan, varname=None, logger=None):
        """
        Appends (i.e. inserts at the end) one or several variable(s).

        Equivalent to `insert_var(..., ind=-1)`.

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of the new variable(s); the size of the array must be

            - a multiple of the number of grid cells (i.e. `.nx * .ny * .nz`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all grid cells

        varname : str or 1D array-like of strs, optional
            name(s) of the new variable(s);
            by default (`None`): variable names are set to "V<num>", where <num>
            starts from the number of variables before the insertion

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        # fname = 'append_var'

        self.insert_var(val=val, varname=varname, ind=self.nv, logger=logger)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=None, indList=None, logger=None):
        """
        Removes variable(s) of given index(es).

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the variable(s) to be removed

        indList : int or 1D array-like of ints
            deprecated (used in place of `ind` if `ind=None`)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'remove_var'

        if ind is None:
            ind = indList
            if ind is None:
                return None

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return None

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ind = np.setdiff1d(np.arange(self.nv), ind)

        self.extract_var(ind, logger=logger)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """
        Removes all variables.
        """
        # fname = 'remove_allvar'

        # Update val array
        del (self.val)
        self.val = np.zeros((0, self.nz, self.ny, self.nx))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, val=np.nan, varname=None, ind=-1, logger=None):
        """
        Sets values and name of one variable (of given index).

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of variable to be set; the size of the array must be

            - a multiple of the number of grid cells (i.e. `.nx * .ny * .nz`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all grid cells

        varname : str, optional
            name of the variable to be set

        ind : int, default: -1
            index of the variable to be set (negative integer for indexing from
            the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'set_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size != self.nxyz():
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.nz, self.ny, self.nx)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, ind=None, indList=None, logger=None):
        """
        Extracts variable(s) (of given index(es)).

        May be used for reordering / duplicating variables.

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the variable(s) to be extracted (kept);
            note: use `ind=[]` to remove all variables

        indList : int or 1D array-like of ints
            deprecated (used in place of `ind` if `ind=None`)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'extract_var'

        if ind is None:
            ind = indList
            if ind is None:
                err_msg = f'{fname}: no index given'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allvar()
            return None

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Update val array
        self.val = self.val[ind,...]

        # Update varname list
        self.varname = [self.varname[i] for i in ind]

        # Update nv
        self.nv = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique_one_var(self, ind=0, ignore_missing_value=True, logger=None):
        """
        Gets unique values of one variable (of given index).

        Parameters
        ----------
        ind : int, default: 0
            index of the variable for which the unique values are retrieved

        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Returns
        -------
        unique_val : 1D array
            unique values of the variable
        """
        fname = 'get_unique_one_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        uval = np.unique(self.val[ii])

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True, ignore_missing_value=True, logger=None):
        """
        Gets proportions (density or count) of unique values of one variable (of given index).

        Parameters
        ----------
        ind : int, default: 0
            index of the variable for which the proportions are retrieved

        density : bool, default: True
            - if `True`: density (proportions) is retrieved
            - if False: counts (number of occurrences) are retrieved

        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Returns
        -------
        unique_val: 1D array
            unique values of the variable

        prop: 1D array
            density (proportions) or counts of the unique values of the variable
        """
        fname = 'get_prop_one_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        uv, cv = np.unique(self.val[ii], return_counts=True)

        if ignore_missing_value:
            ind_known = ~np.isnan(uv)
            uv = uv[ind_known]
            cv = cv[ind_known]

        if density:
            cv = cv / np.sum(cv)

        return uv, cv
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique(self, ignore_missing_value=True):
        """
        Gets unique values over all the variables.

        Parameters
        ----------
        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        Returns
        -------
        unique_val : 1D array
            unique values over all the variables
        """
        # fname = 'get_unique'

        uval = np.unique(self.val)

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop(self, density=True, ignore_missing_value=True, logger=None):
        """
        Gets proportions (density or count) of unique values for each variable.

        Parameters
        ----------
        density : bool, default: True
            - if `True`: density (proportions) is retrieved
            - if `False`: counts (number of occurrences) are retrieved

        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Returns
        -------
        unique_val: 1D array
            unique values of the variable

        prop: 2D array
            density (proportions) or counts of the values `unique_val` for
            each variable:

            - `prop[i, j]`: proportion or count of value `unique_val[j]` for \
            the variable `i`
        """
        # fname = 'get_prop'

        uv_all = self.get_unique(ignore_missing_value)
        n = len(uv_all)
        cv_all = np.zeros((self.nv, n))

        uv_all_ind_nan = None # index in uv_all filled with nan
        if not ignore_missing_value:
            uv_all_ind_nan = np.where(np.isnan(uv_all))[0]
            if uv_all_ind_nan.size:
                uv_all_ind_nan = uv_all_ind_nan[0]
            else:
                uv_all_ind_nan = None

        for i in range(self.nv):
            uv, cv = self.get_prop_one_var(ind=i, density=density, ignore_missing_value=ignore_missing_value, logger=logger)
            for j, v in enumerate(uv):
                if np.isnan(v):
                    cv_all[i, uv_all_ind_nan] = cv[j]
                else:
                    cv_all[i, uv_all==v] = cv[j]

        return uv_all, cv_all
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipx(self):
        """
        Flips variable values according to x axis.
        """
        # fname = 'flipx'

        self.val = self.val[:,:,:,::-1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipy(self):
        """
        Flips variable values according to y axis.
        """
        # fname = 'flipy'

        self.val = self.val[:,:,::-1,:]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipz(self):
        """
        Flips variable values according to z axis.
        """
        # fname = 'flipz'

        self.val = self.val[:,::-1,:,:]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipv(self):
        """
        Flips variable values according to v (variable) axis.
        """
        # fname = 'flipv'

        self.val = self.val[::-1,:,:,:]
        self.varname = self.varname[::-1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permxy(self):
        """
        Permutes / swaps x and y axes.

        (Deprecated, use `swap_xy()`.)
        """
        # fname = 'permxy'

        self.swap_xy()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permxz(self):
        """
        Permutes / swaps x and z axes.

        (Deprecated, use `swap_xz()`.)
        """
        # fname = 'permxz'

        self.swap_xz()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permyz(self):
        """
        Permutes / swaps y and z axes.

        (Deprecated, use `swap_yz()`.)
        """
        # fname = 'permyz'

        self.swap_yz()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def swap_xy(self):
        """
        Swaps x and y axes.
        """
        # fname = 'swap_xy'

        self.val = self.val.swapaxes(2, 3)
        self.nx, self.ny = self.ny, self.nx
        self.sx, self.sy = self.sy, self.sx
        self.ox, self.oy = self.oy, self.ox
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def swap_xz(self):
        """
        Swaps x and z axes.
        """
        # fname = 'swap_xz'

        self.val = self.val.swapaxes(1, 3)
        self.nx, self.nz = self.nz, self.nx
        self.sx, self.sz = self.sz, self.sx
        self.ox, self.oz = self.oz, self.ox
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def swap_yz(self):
        """
        Swaps y and z axes.
        """
        # fname = 'swap_yz'

        self.val = self.val.swapaxes(1, 2)
        self.ny, self.nz = self.nz, self.ny
        self.sy, self.sz = self.sz, self.sy
        self.oy, self.oz = self.oz, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_xzy(self):
        """
        Applies transposition: sends original x, y, z axes to x, z, y axes.

        Equivalent to `swap_yz()`.
        """
        # fname = 'transpose_xyz_to_xzy'

        self.val = self.val.transpose((0, 2, 1, 3))
        self.nx, self.ny, self.nz = self.nx, self.nz, self.ny
        self.sx, self.sy, self.sz = self.sx, self.sz, self.sy
        self.ox, self.oy, self.oz = self.ox, self.oz, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_yxz(self):
        """
        Applies transposition: sends original x, y, z axes to y, x, z axes.

        Equivalent to `swap_xy()`.
        """
        # fname = 'transpose_xyz_to_yxz'

        self.val = self.val.transpose((0, 1, 3, 2))
        self.nx, self.ny, self.nz = self.ny, self.nx, self.nz
        self.sx, self.sy, self.sz = self.sy, self.sx, self.sz
        self.ox, self.oy, self.oz = self.oy, self.ox, self.oz
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_yzx(self):
        """
        Applies transposition: sends original x, y, z axes to y, z, x axes.
        """
        # fname = 'transpose_xyz_to_yzx'

        self.val = self.val.transpose((0, 2, 3, 1))
        self.nx, self.ny, self.nz = self.nz, self.nx, self.ny
        self.sx, self.sy, self.sz = self.sz, self.sx, self.sy
        self.ox, self.oy, self.oz = self.oz, self.ox, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_zxy(self):
        """
        Applies transposition: sends original x, y, z axes to z, x, y axes.
        """
        # fname = 'transpose_xyz_to_zxy'

        self.val = self.val.transpose((0, 3, 1, 2))
        self.nx, self.ny, self.nz = self.ny, self.nz, self.nx
        self.sx, self.sy, self.sz = self.sy, self.sz, self.sx
        self.ox, self.oy, self.oz = self.oy, self.oz, self.ox
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_zyx(self):
        """
        Applies transposition: sends original x, y, z axes to z, y, x axes.

        Equivalent to `swap_xz()`.
        """
        # fname = 'transpose_xyz_to_zyx'

        self.val = self.val.transpose((0, 3, 2, 1))
        self.nx, self.ny, self.nz = self.nz, self.ny, self.nx
        self.sx, self.sy, self.sz = self.sz, self.sy, self.sx
        self.ox, self.oy, self.oz = self.oz, self.oy, self.ox
    # ------------------------------------------------------------------------

    def nxyzv(self):
        """
        Returns the size of the array `.val`, i.e. number of variables times number of grid cells.
        """
        return self.nx * self.ny * self.nz * self.nv

    def nxyz(self):
        """
        Returns the number of grid cells.
        """
        return self.nx * self.ny * self.nz

    def nxy(self):
        """
        Returns the number of grid cells in a xy-section.
        """
        return self.nx * self.ny

    def nxz(self):
        """
        Returns the number of grid cells in a xz-section.
        """
        return self.nx * self.nz

    def nyz(self):
        """
        Returns the number of grid cells in a yz-section.
        """
        return self.ny * self.nz

    def xmin(self):
        """
        Returns min coordinate of the grid along x axis.
        """
        return self.ox

    def ymin(self):
        """
        Returns min coordinate of the grid along y axis.
        """
        return self.oy

    def zmin(self):
        """
        Returns min coordinate of the grid along z axis.
        """
        return self.oz

    def xmax(self):
        """
        Returns max coordinate of the grid along x axis.
        """
        return self.ox + self.nx * self.sx

    def ymax(self):
        """
        Returns max coordinate of the grid along y axis.
        """
        return self.oy + self.ny * self.sy

    def zmax(self):
        """
        Returns max coordinate of the grid along z axis.
        """
        return self.oz + self.nz * self.sz

    def x(self):
        """
        Returns 1D array of "unique" x coordinates of the grid cell centers.

        The returned array is of shape (`.nx`, ).
        """
        return self.ox + 0.5 * self.sx + self.sx * np.arange(self.nx)

    def y(self):
        """
        Returns 1D array of "unique" y coordinates of the grid cell centers.

        The returned array is of shape (`.ny`, ).
        """
        return self.oy + 0.5 * self.sy + self.sy * np.arange(self.ny)

    def z(self):
        """
        Returns 1D array of "unique" z coordinates of the grid cell centers.

        The returned array is of shape (`.nz`, ).
        """
        return self.oz + 0.5 * self.sz + self.sz * np.arange(self.nz)

    def xx(self):
        """
        Returns 3D array of x coordinates of the grid cell centers.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: x coordinate of the grid cell center of index
            `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.tile(self.ox + 0.5 * self.sx + self.sx * np.arange(self.nx), self.ny*self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return xx

    def yy(self):
        """
        Returns 3D array of y coordinates of the grid cell centers.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: y coordinate of the grid cell center of index
            `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.tile(np.repeat(self.oy + 0.5 * self.sy + self.sy * np.arange(self.ny), self.nx), self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return yy

    def zz(self):
        """
        Returns 3D array of z coordinates of the grid cell centers.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: z coordinate of the grid cell center of index
            `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.repeat(self.oz + 0.5 * self.sz + self.sz * np.arange(self.nz), self.nx*self.ny).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return zz

    def ix(self):
        """
        Returns 1D array of "unique" index of grid cell along x axis.

        The returned array is of shape (`.nx`, ).
        """
        return np.arange(self.nx)

    def iy(self):
        """
        Returns 1D array of "unique" index of grid cell along y axis.

        The returned array is of shape (`.ny`, ).
        """
        return np.arange(self.ny)

    def iz(self):
        """
        Returns 1D array of "unique" index of grid cell along z axis.

        The returned array is of shape (`.nz`, ).
        """
        return np.arange(self.nz)

    def ixx(self):
        """
        Returns 3D array of index along x axis of the grid cells.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: index along x axis of the grid cell center of
            index `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.tile(np.arange(self.nx), self.ny*self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return xx

    def iyy(self):
        """
        Returns 3D array of index along y axis of the grid cells.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: index along y axis of the grid cell center of
            index `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.tile(np.repeat(np.arange(self.ny), self.nx), self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return yy

    def izz(self):
        """
        Returns 3D array of index along z axis of the grid cells.

        Returns
        -------
        out : 3D array of shape (`.nz`, `.ny`, `.nx`)
            `out[iz, iy, ix]`: index along z axis of the grid cell center of
            index `iz`, `iy`, `ix` along axis z, y, x respectively
        """
        return np.repeat(np.arange(self.nz), self.nx*self.ny).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return zz

    def vmin(self):
        """
        Returns 1D array of min value of each variable, ignoring `numpy.nan` entries.

        The returned array is of shape (`.nv`, ).
        """
        return np.nanmin(self.val.reshape(self.nv,self.nxyz()),axis=1)

    def vmax(self):
        """
        Returns 1D array of max value of each variable, ignoring `numpy.nan` entries.

        The returned array is of shape (`.nv`, ).
        """
        return np.nanmax(self.val.reshape(self.nv,self.nxyz()),axis=1)
# ============================================================================

# ============================================================================
class PointSet(object):
    """
    Class defining a point set.

    **Attributes**

    npt : int, default: 0
        number of points

    nv : int, default: 0
        number of variables including x, y, z coordinates

    val : 2D array of float of shape (`nv`, `npt`)
        variable values:

        - `val[i, j]`: value of the i-th variable for the j-th point

    varname : list of str, of length `nv`
        variable names:

        - `varname[i]`: name of the i-th variable

    name : str
        name of the point set

    **Methods**
    """
    #
    # Methods
    # -------
    #
    def __init__(self,
                 npt=0,
                 nv=0, 
                 val=np.nan, 
                 varname=None,
                 name="",
                 logger=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        npt : int, default: 0
            number of points

        nv : int, default: 0
            number of variable(s) / attribute(s)

        val : float or array-like of size `nv*npt`
            attribute(s) / variable(s) values

        varname : list of str of length `nv`, optional
            variable names:

            - `varname[iv]`: name of the variable `iv`

            By default (`None`): variable names are set to
            "X", "Y", "Z", "V0", "V1", ...

        name : str, default: ''
            name of the point set

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'PointSet'

        self.npt = int(npt)
        self.nv = int(nv)

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(npt*nv)
        elif valarr.size != npt*nv:
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        self.val = valarr.reshape(nv, npt)

        if varname is None:
            self.varname = []

            if nv > 0:
                self.varname.append('X')

            if nv > 1:
                self.varname.append('Y')

            if nv > 2:
                self.varname.append('Z')

            if nv > 3:
                for i in range(3, nv):
                    self.varname.append(f'V{i-3:d}')

        else:
            varname = list(np.asarray(varname).reshape(-1))
            if len(varname) != nv:
                err_msg = f'{fname}: `varname` has not an acceptable size'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            self.varname = list(np.asarray(varname).reshape(-1))

        self.name = name

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** PointSet object ***'
        out = out + '\n' + "name = '{0.name}'".format(self)
        out = out + '\n' + 'npt = {0.npt} # number of point(s)'.format(self)
        out = out + '\n' + 'nv = {0.nv}  # number of variable(s) (including coordinates)'.format(self)
        out = out + '\n' + 'varname = {0.varname}'.format(self)
        out = out + '\n' + 'val: {0.val.shape}-array'.format(self)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_default_varname(self):
        """
        Sets default variable names: ('X', 'Y', 'Z', 'V0', 'V1', ...).
        """
        # fname = 'set_default_varname'

        self.varname = []

        if self.nv > 0:
            self.varname.append('X')

        if self.nv > 1:
            self.varname.append('Y')

        if self.nv > 2:
            self.varname.append('Z')

        if self.nv > 3:
            for i in range(3, self.nv):
                self.varname.append(f'V{i-3:d}')
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_varname(self, varname=None, ind=-1, logger=None):
        """
        Sets name of the variable of the given index.

        Parameters
        ----------
        varname : str, optional
            name to be set;
            by default (`None`): "V" followed by the variable index is used

        ind : int, default: -1
            index of the variable for which the name is given (negative integer
            for indexing from the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'set_varname'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if varname is None:
            varname = f'V{ii:d}'
        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, val=np.nan, varname=None, ind=0, logger=None):
        """
        Inserts one or several variable(s) at a given index.

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of variable to be set; the size of the array must be

            - a multiple of the number of points (i.e. `.npt`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all points

        varname : str or 1D array-like of strs, optional
            name(s) of the new variable(s);
            by default (`None`): variable names are set to "V<num>", where <num>
            starts from the number of variables before the insertion

        ind : int, default: 0
            index where the new variable(s) is (are) inserted (negative integer
            for indexing from the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'insert_var'

        # Check / set ind
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Check val, set valarr (array of values)
        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size % self.npt != 0:
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        m = valarr.size // self.npt # number of variable to be inserted

        # Check / set varname
        if varname is not None:
            if isinstance(varname, str):
                varname = [varname]
            if (not isinstance(varname, tuple) and not isinstance(varname, list) and not isinstance(varname, np.ndarray)) or len(varname)!=m:
                err_msg = f'{fname}: `varname` does not have an acceptable size'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)
            else:
                varname = list(varname)
        else:
            # set default varname
            varname = [f'V{i:d}' for i in range(self.nv, self.nv+m)]

        # Extend val
        self.val = np.concatenate((self.val[0:ii,...],
                                  valarr.reshape(-1, self.npt),
                                  self.val[ii:,...]),
                                  0)
        # Extend varname list
        self.varname = self.varname[:ii] + varname + self.varname[ii:]

        # Update nv
        self.nv = self.nv + m
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def append_var(self, val=np.nan, varname=None, logger=None):
        """
        Appends (i.e. inserts at the end) one or several variable(s).

        Equivalent to `insert_var(..., ind=-1)`.

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of variable to be set; the size of the array must be

            - a multiple of the number of points (i.e. `.npt`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all points

        varname : str or 1D array-like of strs, optional
            name(s) of the new variable(s);
            by default (`None`): variable names are set to "V<num>", where <num>
            starts from the number of variables before the insertion

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        # fname = 'append_var'

        self.insert_var(val=val, varname=varname, ind=self.nv, logger=logger)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=None, indList=None, logger=None):
        """
        Removes variable(s) of given index(es).

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the variable(s) to be removed

        indList : int or 1D array-like of ints
            deprecated (used in place of `ind` if `ind=None`)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'remove_var'

        if ind is None:
            ind = indList
            if ind is None:
                return None

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return None

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ind = np.setdiff1d(np.arange(self.nv), ind)

        self.extract_var(ind, logger=logger)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """
        Removes all variables.
        """
        # fname = 'remove_allvar'

        # Update val array
        self.val = np.zeros((0, self.npt))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, val=np.nan, varname=None, ind=-1, logger=None):
        """
        Sets values and name of one variable (of given index).

        Parameters
        ----------
        val : float or array-like, default: `numpy.nan`
            value(s) of variable to be set; the size of the array must be

            - a multiple of the number of points (i.e. `.npt`)
            - or 1 (a float is considered of size 1); in this case the value \
            is duplicated once over all points

        varname : str, optional
            name of the variable to be set

        ind : int, default: -1
            index of the variable to be set (negative integer for indexing from
            the end)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'set_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            err_msg = f'{fname}: `val` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.npt)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, ind=None, indList=None, logger=None):
        """
        Extracts variable(s) (of given index(es)).

        May be used for reordering / duplicating variables.

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the variable(s) to be extracted (kept);
            note: use `ind=[]` to remove all variables

        indList : int or 1D array-like of ints
            deprecated (used in place of `ind` if `ind=None`)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'extract_var'

        if ind is None:
            ind = indList
            if ind is None:
                err_msg = f'{fname}: no index given'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allvar()
            return None

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Update val array
        self.val = self.val[ind,...]

        # Update varname list
        self.varname = [self.varname[i] for i in ind]

        # Update nv
        self.nv = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_point(self, ind=None, logger=None):
        """
        Removes point(s) (of given index-es).

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the point(s) to be removed

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'remove_point'

        if ind is None:
            return None

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return None

        ind[ind<0] = self.npt + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.npt)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ind = np.setdiff1d(np.arange(self.npt), ind)

        self.extract_point(ind, logger=logger)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allpoint(self):
        """
        Removes all points.
        """
        # fname = 'remove_allpoint'

        # Update val array
        self.val = np.zeros((self.nv, 0))

        # Update npt
        self.npt = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_uninformed_point(self):
        """
        Removes point(s) where all variables are undefined (`numpy.nan`).
        """
        # fname = 'remove_uninformed_point'

        # Get index of variables that are not coordinates
        ind = np.where([not (self.varname[i] in ('x', 'X', 'y', 'Y', 'z', 'Z')) for i in range(self.nv)])

        # Remove uninformed points
        self.val = self.val[:, ~np.all(np.isnan(self.val[ind]), axis=0)]

        # Update npt
        self.npt = self.val.shape[1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_point(self, ind=None, logger=None):
        """
        Extracts point(s) (of given index-es).

        May be used for reordering / duplicating points.

        Parameters
        ----------
        ind : int or 1D array-like of ints
            index(es) of the point(s) to be extracted (kept);
            note: use `ind=[]` to remove all points

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'extract_point'

        if ind is None:
            err_msg = f'{fname}: no index given'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allpt()
            return None

        ind[ind<0] = self.npt + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.npt)):
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Update val array
        self.val = self.val[:, ind]

        # Update npt
        self.npt = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique_one_var(self, ind=0, ignore_missing_value=True, logger=None):
        """
        Gets unique values of one variable (of given index).

        Parameters
        ----------
        ind : int, default: 0
            index of the variable for which the unique values are retrieved

        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Returns
        -------
        unique_val : 1D array
            unique values of the variable
        """
        fname = 'get_unique_one_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        uval = np.unique(self.val[ii])

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True, ignore_missing_value=True, logger=None):
        """
        Gets proportions (density or count) of unique values of one variable (of given index).

        Parameters
        ----------
        ind : int, default: 0
            index of the variable for which the proportions are retrieved

        density : bool, default: True
            - if `True`: density (proportions) is retrieved
            - if `False`: counts (number of occurrences) are retrieved

        ignore_missing_value : bool, default: True
            - if `True`: missing values (`numpy.nan`) are ignored (if present)
            - if `False`: value `numpy.nan` is retrieved in output if present

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Returns
        -------
        unique_val: 1D array
            unique values of the variable

        prop: 1D array
            density (proportions) or counts of the unique values of the variable
        """
        fname = 'get_prop_one_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            err_msg = f'{fname}: invalid index'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        uv, cv = np.unique(self.val[ii], return_counts=True)

        if ignore_missing_value:
            ind_known = ~np.isnan(uv)
            uv = uv[ind_known]
            cv = cv[ind_known]

        if density:
            cv = cv / np.sum(cv)

        return uv, cv
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def to_dict(self):
        """
        Returns the point set as a dictionary.
        """
        return {name: values for name, values in zip(self.varname, self.val)}
    # ------------------------------------------------------------------------

    def x(self):
        """
        Returns 1D array of x coordinate of the points (assuming stored in variable index 0).

        The returned array is `.var[0]`.
        """
        return self.val[0]

    def y(self):
        """
        Returns 1D array of y coordinate of the points (assuming stored in variable index 1).

        The returned array is `.var[1]`.
        """
        return self.val[1]

    def z(self):
        """
        Returns 1D array of z coordinate of the points (assuming stored in variable index 2).

        The returned array is `.var[2]`.
        """
        return self.val[2]

    def xmin(self):
        """
        Returns min x coordinate of the points (assuming stored in variable index 0).
        """
        return np.min(self.val[0])

    def ymin(self):
        """
        Returns min y coordinate of the points (assuming stored in variable index 1).
        """
        return np.min(self.val[1])

    def zmin(self):
        """
        Returns min z coordinate of the points (assuming stored in variable index 2).
        """
        return np.min(self.val[2])

    def xmax(self):
        """
        Returns max x coordinate of the points (assuming stored in variable index 0).
        """
        return np.max(self.val[0])

    def ymax(self):
        """
        Returns max y coordinate of the points (assuming stored in variable index 1).
        """
        return np.max(self.val[1])

    def zmax(self):
        """
        Returns max z coordinate of the points (assuming stored in variable index 2).
        """
        return np.max(self.val[2])
# ============================================================================

# ============================================================================
class Img_interp_func(object):
    """
    Class defining an interpolator (function) of one variable in an image.

    The class is callable, an interpolator (function) with one argument, a
    2D array, each line defining the coordinates of a point where the
    interpolation is done; the number of columns (coordinates) is equal to the
    number of "no slice" axes (see parameters `ix`, `iy`, `iz` below), each
    column being the coordinate in the corresponding axis direction.

    **Attributes**

    im : :class:`Img`
        image (3D grid with attached variable(s))

    ind : int, default: 0
        index of the variable to be interpolated

    ix : int or `None` (default)
        - if not given (`None`), no slice, all values in the array of the \
        variable values along the x axis are considered, and coordinate along x \
        axis will be required for points passed to the interpolator
        - if given (not `None`): slice index along x axis, only the given slice \
        corresponding to x axis in the array of the variable values is considered, \
        and coordinate along x axis will not be considered for points passed to \
        the interpolator

    iy : int or `None` (default)
        same as `ix`, but for y axis

    iz : int or `None` (default)
        same as `ix`, but for z axis

    angle_var : bool, default: False
        - if `True`: variable to be interpolated are considered as angles, and the \
        interpolation is done by first interpolating the cosine and the sine of \
        the angle values and then by retrieving the corresponding angle (by \
        using the function `numpy.arctan2`)
        - if `False`: values are interpolated directly

    angle_deg : bool, default: True
        used if `angle_var=True`:

        - if `True`: the variable values are angles in degrees
        - if `False`: the variable values are angles in radians

    order : int, default: 1
        order for the interpolator within the domain of the image grid,
        integer in {0, ..., 5}
        (1: linear, 3: cubic, 5: quintic)

    mode : str, default: 'nearest'
        determines the behaviour of the interpolator beyond the domain of the
        image grid

    cval : float, default: `numpy.nan`
        value used for evaluation beyond the domain of the image grid, used if
        `mode=constant`

    Notes
    -----
    See web page
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html
    under "Uniformly space data", introducing a similar class originating
    from the Johanness Buchner's 'regulargrid' package on
    https://github.com/JohannesBuchner/regulargrid/

    Examples
    --------
        >>> # Define an image
        >>> nx, ny, nz = 4, 5, 6
        >>> sx, sy, sz = 1.0, 1.0, 1.0
        >>> ox, oy, oz = 0.0, 0.0, 0.0
        >>> im = Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz,
        >>>          nv=1, val=np.arange(nx*ny*nz))
        >>>
        >>> # Define an interpolator
        >>> interp = Img_interp_func(im)
        >>>
        >>> # Evaluate the interpolator on some points
        >>> points = np.array([[2.2, 3.4, 1.2], [2.7, 4.1, 5.2]])
        >>> v = interp(points)
        >>>
        >>> interp2 = scipy.interpolate.RegularGridInterpolator(
        >>>     (im.x(), im.y(), im.z()), im.val[0].transpose(2, 1, 0),
        >>>     method='linear', bounds_error=False, fill_value=np.nan)
        >>> v2 = interp2(points) # gives same values except for points beyond
        >>>                      # the domain of the image grid

    **Methods**
    """
    def __init__(self,
                 im,
                 ind=0, 
                 ix=None, 
                 iy=None,
                 iz=None,
                 angle_var=False, 
                 angle_deg=True,
                 order=1, 
                 mode='nearest', 
                 cval=np.nan, 
                 logger=None):
        """
        Inits an instance of the class (interpolator function).

        Parameters
        ----------
        im : :class:`Img`
            image (3D grid with attached variable(s))

        ind : int, default: 0
            index of the variable to be interpolated

        ix : int or None (default)
            - if not given (`None`), no slice, all values in the array of the \
            variable values along the x axis are considered, and coordinate along x \
            axis will be required for points passed to the interpolator
            - if given (not `None`): slice index along x axis, only the given slice \
            corresponding to x axis in the array of the variable values is considered, \
            and coordinate along x axis will not be considered for points passed to \
            the interpolator

        iy : int or None (default)
            same as `ix`, but for y axis

        iz : int or None (default)
            same as `ix`, but for z axis

        angle_var: bool, default: False
            - if `True`: variable to be interpolated are considered as angles, and the \
            interpolation is done by first interpolating the cosine and the sine of \
            the angle values and then by retrieving the corresponding angle (by \
            using the function `numpy.arctan2`)
            - if `False`: values are interpolated directly

        angle_deg: bool, default: True
            used if `angle_var=True`:

            - if `angle_deg=True`: the variable values are angles in degrees
            - if `angle_deg=False`: the variable values are angles in radians

        order : int, default 1
            order for the interpolator within the domain of the image grid
            (1: linear, 3: cubic, 5: quintic)

        mode : str, default 'nearest'
            determines the behaviour of the interpolator beyond the domain of the
            image grid

        cval : float, default `numpy.nan`
            value used for evaluation beyond the domain of the image grid, used if
            `mode=constant`

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'Img_interp_func'

        # Check image
        if not isinstance(im, Img):
            err_msg = f'{fname}: `im` is not a geone image'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Check set variable index
        if ind < 0:
            iv = im.nv + ind
        else:
            iv = ind

        if iv < 0 or iv >= im.nv:
            err_msg = f'{fname}: invalid variable index (`ind`)'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Check index along x axis
        if ix is not None:
            if ix < 0:
                ix = im.nx + ix

            if ix < 0 or ix >= im.nx:
                err_msg = f'{fname}: invalid index for x axis (`ix`)'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        # Check index along y axis
        if iy is not None:
            if iy < 0:
                iy = im.ny + iy

            if iy < 0 or iy >= im.ny:
                err_msg = f'{fname}: invalid index for y axis (`iy`)'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        # Check index along z axis
        if iz is not None:
            if iz < 0:
                iz = im.nz + iz

            if iz < 0 or iz >= im.nz:
                err_msg = f'{fname}: invalid index for z axis (`iz`)'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        # Get array of values, spacing, and minimal coordinates for the interpolator
        if ix is None:
            if iy is None:
                if iz is None:
                    # interpolator along x, y, z axes
                    values = im.val[iv, :, :, :].transpose(2, 1, 0)
                    spacing = np.array([im.sx, im.sy, im.sz])
                    min_coords = np.array([im.ox, im.oy, im.oz]) + 0.5*spacing
                else:
                    # interpolator along x, y axes
                    values = im.val[iv, iz, :, :].transpose(1, 0)
                    spacing = np.array([im.sx, im.sy])
                    min_coords = np.array([im.ox, im.oy]) + 0.5*spacing
            else:
                if iz is None:
                    # interpolator along x, z axes
                    values = im.val[iv, :, iy, :].transpose(1, 0)
                    spacing = np.array([im.sx, im.sz])
                    min_coords = np.array([im.ox, im.oz]) + 0.5*spacing
                else:
                    # interpolator along x axis
                    values = im.val[iv, iz, iy, :]
                    spacing = np.array([im.sx])
                    min_coords = np.array([im.ox]) + 0.5*spacing
        else:
            if iy is None:
                if iz is None:
                    # interpolator along y, z axes
                    values = im.val[iv, :, :, ix].transpose(1, 0)
                    spacing = np.array([im.sy, im.sz])
                    min_coords = np.array([im.oy, im.oz]) + 0.5*spacing
                else:
                    # interpolator along y axis
                    values = im.val[iv, iz, :, ix]
                    spacing = np.array([im.sy])
                    min_coords = np.array([im.oy]) + 0.5*spacing
            else:
                if iz is None:
                    # interpolator along z axis
                    values = im.val[iv, :, iy, ix]
                    spacing = np.array([im.sz])
                    min_coords = np.array([im.oz]) + 0.5*spacing
                else:
                    err_msg = f'{fname}: none of the axes corresponds to "no slice"'
                    if logger: logger.error(err_msg)
                    raise ImgError(err_msg)

        self.angle_var = angle_var
        self.angle_deg = angle_deg
        if angle_var and angle_deg:
            # Transform values (angles) in radians
            values = np.pi/180.0*values
        self.values = values
        self.spacing = spacing
        self.min_coords = min_coords
        self.order = order
        self.mode = mode
        self.cval = cval

    # def __call__(self, points):
    #     """
    #     Returns the evaluation of the interpolation at the given points.
    #
    #     Parameters
    #     ----------
    #     points : 2D array (or 1D array-like)
    #         each row is a point where the interpolation is done, the columns
    #         correspond to the coordinates along the "no sliced" axes (see doc of
    #         the class); notes, with d is the number of "no sliced" axes
    #         (dimension):
    #         * 1D array-like of size d is accepted for the evaluation at one point
    #         * if d=1: 1D array-like of size m is accepted for the evaluation at
    #         m points
    #     """
    #     points = np.atleast_2d(points)
    #
    #     if self.values.ndim == 1:
    #         points = points.reshape(-1, 1)
    #
    #     # Convert points coordinates to grid (pixel) coordinates
    #     grid_coords = (points - self.min_coords)/self.spacing
    #
    #     # Do interpolation
    #     return scipy.ndimage.map_coordinates(self.values, grid_coords.T, order=self.order, mode=self.mode, cval=self.cval)

    def __call__(self, points):
        """
        Interpolates a variable (defined on an image grid) at given points.

        Parameters
        ----------
        points : 2D array (or 1D array-like)
            each row is a point where the interpolation is done, the columns
            correspond to the coordinates along the "no sliced" axes (see doc of
            the class); notes, with d is the number of "no sliced" axes
            (dimension):

            * 1D array-like of size d is accepted for the evaluation at one point
            * if d=1: 1D array-like of size m is accepted for the evaluation at \
            m points

        Returns
        -------
        y : 1D array
            evaluation of the variable at `points`
        """
        points = np.atleast_2d(points)

        if self.values.ndim == 1:
            points = points.reshape(-1, 1)

        # Convert points coordinates to grid (pixel) coordinates
        grid_coords = (points - self.min_coords)/self.spacing

        # Do interpolation
        if self.angle_var:
            # Interpolation of cosine and sine of the values (already in radians)
            cos = scipy.ndimage.map_coordinates(np.cos(self.values), grid_coords.T, order=self.order, mode=self.mode, cval=self.cval)
            sin = scipy.ndimage.map_coordinates(np.sin(self.values), grid_coords.T, order=self.order, mode=self.mode, cval=self.cval)
            y = np.arctan2(sin, cos)
            if self.angle_deg:
                # Transform the result in degree
                y = 180.0/np.pi*y
            return y
        else:
            return scipy.ndimage.map_coordinates(self.values, grid_coords.T, order=self.order, mode=self.mode, cval=self.cval)
# ============================================================================

# ----------------------------------------------------------------------------
def copyImg(im, varInd=None, varIndList=None, logger=None):
    """
    Copies an image, with all or a subset of variables.

    Parameters
    ----------
    im : :class:`Img`
        input image (3D grid with attached variables)

    varInd : int or 1D array-like of ints, or None (default)
        index(es) of the variables to be copied, use `varInd=[]` to copy only
        the grid geometry; by default (`None`): all variables are copied"

    varIndList : int or 1D array-like of ints, or None (default)
        deprecated (used in place of `varInd` if `varInd=None`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        a copy of the input image (not a reference to) with the specified
        variable(s).
    """
    fname = 'copyImg'

    if varInd is None:
        varInd = varIndList

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if varInd.size == 0:
            # empty list of variable
            im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                         sx=im.sx, sy=im.sy, sz=im.sz,
                         ox=im.ox, oy=im.oy, oz=im.oz,
                         nv=0, name=im.name,
                         logger=logger)
        else:
            # Check if each index is valid
            if np.sum([iv in range(im.nv) for iv in varInd]) != len(varInd):
                err_msg = f'{fname}: invalid index-es'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                         sx=im.sx, sy=im.sy, sz=im.sz,
                         ox=im.ox, oy=im.oy, oz=im.oz,
                         nv=len(varInd), val=im.val[varInd], varname=[im.varname[i] for i in varInd],
                         name=im.name,
                         logger=logger)
            # im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
            #              sx=im.sx, sy=im.sy, sz=im.sz,
            #              ox=im.ox, oy=im.oy, oz=im.oz,
            #              nv=len(varInd),
            #              name=im.name,
            #              logger=logger)
            # for i, iv in enumerate(varInd):
            #     im_out.set_var(val=im.val[iv,...], varname=im.varname[iv], ind=i)
    else:
        # Copy all variables
        im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                     sx=im.sx, sy=im.sy, sz=im.sz,
                     ox=im.ox, oy=im.oy, oz=im.oz,
                     nv=im.nv, val=np.copy(im.val), varname=list(np.copy(np.asarray(im.varname))),
                     name=im.name,
                     logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def copyPointSet(ps, varInd=None, varIndList=None, logger=None):
    """
    Copies a point set, with all or a subset of variables.

    Parameters
    ----------
    ps : :class:`PointSet`
        input point set

    varInd : int or 1D array-like of ints, or None (default)
        index(es) of the variables to be copied, use `varInd=[]` to copy only
        the grid geometry; by default (`None`): all variables are copied"

    varIndList : int or 1D array-like of ints, or None (default)
        deprecated (used in place of `varInd` if `varInd=None`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps_out : :class:`PointSet`
        a copy of the input point set (not a reference to) with the specified
        variable(s).
    """
    fname = 'copyPointSet'

    if varInd is None:
        varInd = varIndList

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        # Check if each index is valid
        if np.sum([iv in range(ps.nv) for iv in varInd]) != len(varInd):
            err_msg = f'{fname}: invalid index-es'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ps_out = PointSet(npt=ps.npt,
                          nv=len(varInd), val=ps.val[varInd], varname=[ps.varname[i] for i in varInd],
                          name=ps.name)
    else:
        # Copy all variables
        ps_out = PointSet(npt=ps.npt,
                          nv=ps.nv, val=np.copy(ps.val), varname=list(np.copy(np.asarray(ps.varname))),
                          name=ps.name)

    return ps_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pointToGridIndex(
        x, y, z,
        sx=1.0, sy=1.0, sz=1.0,
        ox=0.0, oy=0.0, oz=0.0):
    """
    Converts float point coordinates to index grid.

    Parameters
    ----------
    x : float or 1D array-like of floats
        x coordinate of point(s)

    y : float or 1D array-like of floats
        y coordinate of point(s)

    z : float or 1D array-like of floats
        z coordinate of point(s)

        Note: `x`, `y`, `z` are of same size

    sx : float, default: 1.0
        cell size along x axis

    sy : float, default: 1.0
        cell size along y axis

    sz : float, default: 1.0
        cell size along z axis

        Note: `(sx, sy, sz)` is the cell size

    ox : float, default: 0.0
        origin of the grid along x axis (x coordinate of cell border)

    oy : float, default: 0.0
        origin of the grid along y axis (y coordinate of cell border)

    oz : float, default: 0.0
        origin of the grid along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    Returns
    -------
    ix : float or 1D array
        grid node index along x axis for each input points

    iy : float or 1D array
        grid node index along y axis for each input points

    iz : float or 1D array
        grid node index along z axis for each input points

    Notes
    -----
    Warning: no check is done if the input point(s) is (are) within the grid.
    """
    # fname = 'pointToGridIndex'

    # Get node index (nearest node)
    c = np.array((np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z))).T
    jc = (c - np.array([ox, oy, oz]))/np.array([sx, sy, sz])
    ic = jc.astype(int)

    # Round to lower index if between two grid node and index is positive
    ic = ic - 1 * np.all((ic == jc, ic > 0), axis=0)

    ix = ic[:, 0]
    iy = ic[:, 1]
    iz = ic[:, 2]

    # Set ix (resp. iy, iz) as int if x (resp. y, z) is float (or int) in input
    if np.asarray(x).ndim == 0:
        ix = ix[0]
    if np.asarray(y).ndim == 0:
        iy = iy[0]
    if np.asarray(z).ndim == 0:
        iz = iz[0]

    return ix, iy, iz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def gridIndexToSingleGridIndex(
        ix, iy, iz,
        nx, ny, nz):
    """
    Converts grid index(es) into single grid cell index(es).

    Parameters
    ----------
    ix : int or 1D array of ints
        grid cell index(s) along x axis

    iy : int or 1D array of ints
        grid cell index(s) along y axis

    iz : int or 1D array of ints
        grid cell index(s) along z axis

        Note: `ix`, `iy`, `iz` are of same size

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis (unused)

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    Returns
    -------
    i : int or 1D array
        single grid cell index (`ix + nx*iy + nx*ny*iz`) of input indexes
    """
    # fname = 'gridIndexToSingleGridIndex'

    return ix + nx * (iy + ny * iz)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def singleGridIndexToGridIndex(i, nx, ny, nz):
    """
    Converts single grid cell index(es) into grid cell index(es) along each axis.

    Parameters
    ----------
    i : int or 1D array of ints
        single grid index

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis (unused)

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    Returns
    -------
    ix : float or 1D array of ints
        grid cell index(s) along x axis of input single grid index(es)

    iy : float or 1D array of ints
        grid cell index(s) along y axis of input single grid index(es)

    iz : float or 1D array of ints
        grid cell index(s) along z axis of input single grid index(es)
    """
    # fname = 'singleGridIndexToGridIndex'

    nxy = nx*ny
    iz = i//nxy
    j = i%nxy
    iy = j//nx
    ix = j%nx

    return ix, iy, iz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageToPointSet(im, remove_uninformed_cell=True):
    """
    Converts an image into a point set.

    Parameters
    ----------
    im : :class:`Img`
        input image (3D grid with attached variables)

    remove_uninformed_cell : bool, default: True
        - if `True`: image grid cells with no value, i.e. with all variable values \
        missing (`numpy.nan`), are not considered in the output point set
        - if `False`: every image grid cell are considered in the output point set

    Returns
    -------
    ps : :class:`PointSet`
        point set corresponding to the input image, the 3 first variables are the
        x, y, z coordinates of the grid cell centers, the next variable are the
        variables from the input image
    """
    # fname = 'imageToPointSet'

    if remove_uninformed_cell:
        ind_known = ~np.all(np.isnan(im.val), axis=0)
        ps_val = np.vstack((
                    im.xx()[ind_known],
                    im.yy()[ind_known],
                    im.zz()[ind_known],
                    im.val[:, ind_known]))
    else:
        ps_val = np.vstack((im.xx(), im.yy(), im.zz(), im.val.reshape(im.nv, -1)))

    # Initialize point set
    ps = PointSet(npt=ps_val.shape[1], nv=ps_val.shape[0], val=ps_val,
                  varname=['X', 'Y', 'Z'] + im.varname)

    return ps
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def aggregateDataPointsWrtGrid(
        x, y, z, v,
        nx, ny, nz,
        sx=1.0, sy=1.0, sz=1.0,
        ox=0.0, oy=0.0, oz=0.0,
        op='mean',
        return_inverse=False,
        verbose=0,
        logger=None,
        **kwargs):
    """
    Aggregates points in same cells of a given grid geometry.

    The points out of the grid (defined with given parameters) are removed and
    the points falling in a same grid cell are aggregated by taking the mean
    coordinates and by applying the operation `op` for the value of each variable.

    Parameters
    ----------
    x : float or 1D array-like of floats
        x coordinate of point(s)

    y : float or 1D array-like of floats
        y coordinate of point(s)

    z : float or 1D array-like of floats
        z coordinate of point(s)

        Note: `x`, `y`, `z` are of same size

    v : float or 1D array-like or 2D array-like of floats
        values attached to point(s), each row (if 2D array) corresponds to a
        same variable ; last dimension of same size as `x`, `y`, `z`

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis (unused)

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    sx : float, default: 1.0
        cell size along x axis

    sy : float, default: 1.0
        cell size along y axis

    sz : float, default: 1.0
        cell size along z axis

        Note: `(sx, sy, sz)` is the cell size

    ox : float, default: 0.0
        origin of the grid along x axis (x coordinate of cell border)

    oy : float, default: 0.0
        origin of the grid along y axis (y coordinate of cell border)

    oz : float, default: 0.0
        origin of the grid along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    op : str {'min', 'max', 'mean', 'std', 'var', 'quantile', 'most_freq', 'random'}, default: 'mean'
        operation used to aggregate values of data points falling in a same grid
        cell:

        - if `op='most_freq'`: most frequent value is selected (smallest one if \
        more than one value with the maximal frequence)
        - if `op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<op>` is used with the additional \
        parameters given by `kwargs`, note that, e.g. `op='quantile'` requires \
        the additional parameter `q=<quantile_to_compute>`

    return_inverse : bool, default: False
        if `True`, return the index of the aggregated point corresponding to each
        input point (see Returns below)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to `numpy.<op>` function, e.g.
        `ddof=1` if `op='std'` or`op='var'`

    Returns
    -------
    x_out : 1D array of floats
        x coordinate of aggregated point(s)

    y_out : 1D array of floats
        y coordinate of aggregated point(s)

    z_out : 1D array of floats
        z coordinate of aggregated point(s)

    v_out : 1D array or 2D array of floats
        values attached to aggregated point(s), each row (if 2D array)
        corresponds to a same variable

    i_inv : 1D array of ints, optional
        indexes of the aggregated points, array of ints of same size as
        `x, `y`, `z`, `v` (last dimension) and such that the i-th data point
        `((x[i], y[i], z[i]), v[..., i])`
        contributes to the `i_inv[i]`-th aggregated point
        `((x_out[i_inv[i]], y_out[i_inv[i]], z_out[i_inv[i]]), v_out[..., i_inv[i]])`,
        or `i_inv[i] = -1` if the i-th data point has been removed;
        returned if `return_inverse=True`
    """
    fname = 'aggregateDataPointsWrtGrid'

    if np.asarray(v).ndim > 1:
        multi_var = True
    else:
        multi_var = False

    x = np.atleast_1d(x).reshape(-1)
    y = np.atleast_1d(y).reshape(-1)
    z = np.atleast_1d(z).reshape(-1)
    v = np.atleast_2d(v).reshape(-1, x.size)

    # Keep only the points within the grid
    ind = np.all((x >= ox, x <= ox+sx*nx, y >= oy, y <= oy+sy*ny, z >= oz, z <= oz+sz*nz), axis=0)
    if not np.any(ind):
        # no point in the grid
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        if multi_var:
            v = np.zeros((v.shape[0], 0))
        else:
            v = np.zeros(0)
        if return_inverse:
            return x, y, z, v, -1 * np.ones(len(x), dtype='int')
        return x, y, z, v

    x = x[ind]
    y = y[ind]
    z = z[ind]
    v = v[:, ind].T

    # Get node index (nearest node)
    c = np.array((x, y, z)).T
    jc = (c - np.array([ox, oy, oz]))/np.array([sx, sy, sz])
    ic = jc.astype(int)

    # Round to lower index if between two grid node and index is positive
    ic = ic - 1 * np.all((ic == jc, ic > 0), axis=0)

    ix, iy, iz = ic[:, 0], ic[:, 1], ic[:, 2]
    # ix, iy, iz = pointToGridIndex(x, y, z, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz) # Equivalent
    ic = ix + nx * (iy + ny * iz) # single-indexes

    ic_unique, ic_inv = np.unique(ic, return_inverse=True)
    if len(ic_unique) != len(ic):
        # Aggretation is needed
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: more than one point in the same cell (aggregation operation: {op})!')
            else:
                print(f'{fname}: WARNING: more than one point in the same cell (aggregation operation: {op})!')
        # Prepare operation
        if op == 'max':
            func = np.nanmax
        elif op == 'mean':
            func = np.nanmean
        elif op == 'min':
            func = np.nanmin
        elif op == 'std':
            func = np.nanstd
        elif op == 'var':
            func = np.nanvar
        elif op == 'quantile':
            func = np.nanquantile
            if 'q' not in kwargs:
                err_msg = f"({fname}): keyword argument 'q' required by `op='quantile'`"
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        elif op == 'most_freq':
            def func(arr, axis=0):
                # fake keyword argument `axis=0`, because the function can be called with it below
                arr_out = np.zeros(arr.shape[1])
                for i in range(arr.shape[1]):
                    arr_unique, arr_count = np.unique(arr[:, i], return_counts=True)
                    arr_out[i] = arr_unique[np.argmax(arr_count)]
                return arr_out
        elif op == 'random':
            def func(arr, axis=0):
                # fake keyword argument `axis=0`, because the function can be called with it below
                return arr[np.random.randint(arr.shape[0])]
        else:
            err_msg = f"({fname}): unkown operation '{op}'"
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        c = np.array([c[ic_inv==j].mean(axis=0) for j in range(len(ic_unique))])
        v = np.array([func(np.asarray(v[ic_inv==j]), axis=0, **kwargs) for j in range(len(ic_unique))])

    else:
        # Reorder points
        c = np.array([c[ic_inv==j][0] for j in range(len(ic_unique))])
        v = np.array([v[ic_inv==j][0] for j in range(len(ic_unique))])

    x, y, z = c.T # unpack
    v = v.T
    if not multi_var:
        v = v[0]

    if return_inverse:
        i_inv = np.zeros(len(ic), dtype='int')
        for j in range(len(ic_unique)):
            i_inv[ic_inv==j] = j
        for j in np.where(~ind)[0]:
            i_inv = np.insert(i_inv, j, -1)
        return x, y, z, v, i_inv

    return x, y, z, v
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageFromPoints(
        points,
        values=None,
        varname=None,
        nx=None, ny=None, nz=None,
        sx=None, sy=None, sz=None,
        ox=None, oy=None, oz=None,
        xmin_ext=0.0, xmax_ext=0.0,
        ymin_ext=0.0, ymax_ext=0.0,
        zmin_ext=0.0, zmax_ext=0.0,
        indicator_var=False, count_var=False,
        op='mean',
        verbose=0,
        logger=None,
        **kwargs):
    """
    Returns an image from points with attached variables.

    The grid geometry of the output image is set by the given parameters or
    computed from the given point coordinates. The variables attached to grid
    cells are aggregated point values according to the operation `op` (points
    falling in the same grid cells are aggregated). In addition an "indicator"
    variable with value 1 at cells cointaining at least one point (0 elsewhere)
    and a "count" variable indicating the number of point(s) in the cells, can
    also be retrieved.

    The output image grid geometry is defined as follows for the x axis (similar
    for y and z axes):

    - `ox` (origin), `nx` (number of cells) and `sx` (resolution, cell size)
    - or only `nx`: `ox` and `sx` automatically computed
    - or only `sx`: `ox` and `nx` automatically computed

    In the first case, points out of the specified grid are removed.
    In the two last cases, the parameters `xmin_ext`, `xmax_ext`, are used and
    the approximate limit of the grid along x axis is set to x0, x1, where

    - x0: min x coordinate of the points minus `xmin_ext`
    - x1: max x coordinate of the points plus `xmax_ext`

    Note that points in 1D or 2D are accepted, if the points are in 1D, the
    default values

    - `ny=nz=1`, `sy=sz=1.0`, `oy=oz=-0.5` are used

    and if the points are in 2D, the default values

    - `nz=1`, `sz=1.0`, `oz=-0.5` are used

    Parameters
    ----------
    points : 2D array-like of shape (n, d), or 1D array-like of shape (d, )
        each row contains the float coordinates of one point in dimension d
        (1, 2, or 3); note: if 1D array-like, one point at all is given

    values : 1D or 2D array-like, optional
        values attached to point(s), each row of v (if 2D array) corresponds
        to a same variable

    varname : 1D array of str, or str, optional
        variable name(s) (one name per row of `values`), by default: names of
        class `Img` are used

    nx : int, optional
        number of grid cells along x axis; see above for possible inputs

    ny : int, optional
        number of grid cells along y axis; see above for possible inputs

    nz : int, optional
        number of grid cells along z axis; see above for possible inputs

    sx : float, optional
        cell size along x axis; see above for possible inputs

    sy : float, optional
        cell size along y axis; see above for possible inputs

    sz : float, optional
        cell size along z axis; see above for possible inputs

    ox : float, optional
        origin of the grid along x axis (x coordinate of cell border);
        see above for possible inputs

    oy : float, optional
        origin of the grid along y axis (y coordinate of cell border);
        see above for possible inputs

    oz : float, optional
        origin of the grid along z axis (z coordinate of cell border);
        see above for possible inputs

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    xmin_ext : float, default: 0.0
        extension beyond the min x coordinate of the points (see above)

    xmax_ext : float, default: 0.0
        extension beyond the max x coordinate of the points (see above)

    ymin_ext : float, default: 0.0
        extension beyond the min y coordinate of the points (see above)

    ymax_ext : float, default: 0.0
        extension beyond the max y coordinate of the points (see above)

    zmin_ext : float, default: 0.0
        extension beyond the min z coordinate of the points (see above)

    zmax_ext : float, default: 0.0
        extension beyond the max z coordinate of the points (see above)

    indicator_var : bool, default: False
        indicating if the "indicator" variable is added (prepended) (see above)

    count_var : bool, default: False
        indicating if the "count" variable is added (prepended) (see above)

    op : str {'min', 'max', 'mean', 'std', 'var', 'quantile', 'most_freq', 'random'}, default: 'mean'
        operation used to aggregate values of data points falling in a same grid
        cell

        - if `op='most_freq'`: most frequent value is selected (smallest one if \
        more than one value with the maximal frequence)
        - if `op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<op>` is used with the additional \
        parameters given by `kwargs`, note that, e.g. `op='quantile'` requires \
        the additional parameter `q=<quantile_to_compute>`

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to `numpy.<op>` function, e.g.
        `ddof=1` if `op='std'` or`op='var'`

    Returns
    -------
    im : :class:`Img`
        output image (see above)
    """
    fname = 'imageFromPoints'

    points = np.atleast_2d(points)
    d = points.shape[1]
    if d == 0:
        err_msg = f'{fname}: no point given'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if d not in (1, 2, 3):
        err_msg = f'{fname}: `points` of invalid dimension'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Deal with x axis
    # ----------------
    if ox is not None:
        if nx is None or sx is None:
            err_msg = f'{fname}: if `ox` is given, `nx` and `sx` must be given'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        x0, x1 = points[:, 0].min() - xmin_ext, points[:, 0].max() + xmax_ext
        if nx is None and sx is not None:
            nx = int((x1 - x0)/sx) + 1
        elif nx is not None and sx is None:
            sx = (x1 - x0)/nx
        else:
            err_msg = f'{fname}: defining grid (x axis)'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        ox = x0 - 0.5*(nx*sx - (x1-x0))

    # Deal with y axis
    # ----------------
    if d == 1:
        ny, sy, oy = 1, 1.0, -0.5
    elif oy is not None:
        if ny is None or sy is None:
            err_msg = f'{fname}: if `oy` is given, `ny` and `sy` must be given'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        y0, y1 = points[:, 1].min() - ymin_ext, points[:, 1].max() + ymax_ext
        if ny is None and sy is not None:
            ny = int((y1 - y0)/sy) + 1
        elif ny is not None and sy is None:
            sy = (y1 - y0)/ny
        else:
            err_msg = f'{fname}: defining grid (y axis)'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        oy = y0 - 0.5*(ny*sy - (y1-y0))

    # Deal with z axis
    # ----------------
    if d < 3:
        nz, sz, oz = 1, 1.0, -0.5
    elif oz is not None:
        if nz is None or sz is None:
            err_msg = f'{fname}: if `oz` is given, `nz` and `sz` must be given'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        z0, z1 = points[:, 2].min() - zmin_ext, points[:, 2].max() + zmax_ext
        if nz is None and sz is not None:
            nz = int((z1 - z0)/sz) + 1
        elif nz is not None and sz is None:
            sz = (z1 - z0)/nz
        else:
            err_msg = f'{fname}: defining grid (z axis)'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        oz = z0 - 0.5*(nz*sz - (z1-z0))

    # Define output image (without variable)
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=0, logger=logger)

    # Return if no values and no additional variable
    if values is None and not indicator_var and not count_var:
        return im

    # Get grid index of points
    x = points[:, 0]
    if d > 1:
        y = points[:, 1]
    else:
        y = np.zeros_like(x)
    if d > 2:
        z = points[:, 2]
    else:
        z = np.zeros_like(x)

    # Get node index (nearest node)
    c = np.array((x, y, z)).T
    jc = (c - np.array([ox, oy, oz]))/np.array([sx, sy, sz])
    ic = jc.astype(int)

    # Round to lower index if between two grid node and index is positive
    ic = ic - 1 * np.all((ic == jc, ic > 0), axis=0)

    ix, iy, iz = ic[:, 0], ic[:, 1], ic[:, 2]
    # ix, iy, iz = pointToGridIndex(x, y, z, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz) # Equivalent

    # Check if the points are within the grid
    ind = np.all((ix >= 0, ix < nx, iy >= 0, iy < ny, iz >= 0, iz < nz), axis=0)
    if not np.all(ind):
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: point(s) out of the grid')
            else:
                print(f'{fname}: WARNING: point(s) out of the grid')

    # Keep points within the grid
    ix, iy, iz = ix[ind], iy[ind], iz[ind]

    ic = ix + nx * (iy + ny * iz) # single-indexes

    nxyz = nx*ny*nz

    # First, set variable "indicator" and "count" if asked for
    if indicator_var:
        v = np.zeros(nxyz)
        v[ic] = 1.0
        im.append_var(v, varname='indicator', logger=logger)
    if count_var:
        v = np.zeros(nxyz)
        for i in ic:
            v[i] += 1.0
        im.append_var(v, varname='count', logger=logger)

    # Add variable from values
    if values is not None:
        values = np.atleast_2d(values)
        nv = values.shape[0] # number of variables

        # keep points within the grid
        values = values[:, ind].T

        # Prepare array for image variables
        v = np.full((nv, nxyz), np.nan)

        # Aggregate values in grid
        ic_unique, ic_inv = np.unique(ic, return_inverse=True)
        if len(ic_unique) != len(ic):
            # Aggretation is needed
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: more than one point in the same cell (aggregation operation: {op})!')
                else:
                    print(f'{fname}: WARNING: more than one point in the same cell (aggregation operation: {op})!')

            # Prepare operation
            if op == 'max':
                func = np.nanmax
            elif op == 'mean':
                func = np.nanmean
            elif op == 'min':
                func = np.nanmin
            elif op == 'std':
                func = np.nanstd
            elif op == 'var':
                func = np.nanvar
            elif op == 'quantile':
                func = np.nanquantile
                if 'q' not in kwargs:
                    err_msg = f"({fname}): keyword argument 'q' required by `op='quantile'`"
                    if logger: logger.error(err_msg)
                    raise ImgError(err_msg)

            elif op == 'most_freq':
                def func(arr):
                    arr_unique, arr_count = np.unique(arr, return_counts=True)
                    return arr_unique[np.argmax(arr_count)]
            elif op == 'random':
                def func(arr):
                    return arr[np.random.randint(arr.size)]
            else:
                err_msg = f"({fname}): unkown operation '{op}'"
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            values = np.array([func(np.asarray(values[ic_inv==j]), axis=0, **kwargs) for j in range(len(ic_unique))])
            v[:, ic_unique] = values.T
        else:
            v[:, ic] = values.T

        im.append_var(v, varname=varname, logger=logger)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pointSetToImage(
        ps,
        nx=None, ny=None, nz=None,
        sx=None, sy=None, sz=None,
        ox=None, oy=None, oz=None,
        xmin_ext=0.0, xmax_ext=0.0,
        ymin_ext=0.0, ymax_ext=0.0,
        zmin_ext=0.0, zmax_ext=0.0,
        op='mean',
        logger=None,
        **kwargs):
    """
    Converts a point set into an image.

    The first three variable of the point set must correspond to x, y, z
    float coordinates (location of points). Then, it is equivalent to

    - `imageFromPoints(points, values, varname, ..., indicator_var=False, count_var=False, ...)`

    with

    - `points = ps.val[0:3].T`
    - `values=ps.val[3:].T`
    - `varname=ps.varname[3:]`

    where the fisrt parameter, `ps` is an instance of the class `PointSet`.

    See function :func:`imageFromPoints`.
    """
    fname = 'pointSetToImage'

    if ps.nv < 3:
        err_msg = f'{fname}: invalid number of variable (should be > 3)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if ps.varname[0].lower() != 'x' or ps.varname[1].lower() != 'y' or ps.varname[2].lower() != 'z':
        err_msg = f'{fname}: invalid variable: 3 first ones must be x, y, z coordinates'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # points = np.array((ps.x(), ps.y(), ps.z())).T
    im = imageFromPoints(ps.val[:3].T, values=ps.val[3:], varname=ps.varname[3:],
                         nx=nx, ny=ny, nz=nz,
                         sx=sx, sy=sy, sz=sz,
                         ox=ox, oy=oy, oz=oz,
                         xmin_ext=xmin_ext, xmax_ext=xmax_ext,
                         ymin_ext=ymin_ext, ymax_ext=ymax_ext,
                         zmin_ext=zmin_ext, zmax_ext=zmax_ext,
                         indicator_var=False, count_var=False,
                         op=op, 
                         logger=logger,
                         **kwargs)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def isImageDimensionEqual(im1, im2):
    """
    Checks if the grid dimensions of two images are equal.

    Parameters
    ----------
    im1 : :class:`Img`
        first image

    im2 : :class:`Img`
        second image

    Returns
    -------
    bool
        - `True` if number of grid cells along each axis are equal \
        for the two images
        - `False` otherwise
    """
    # fname = 'isImageDimensionEqual'

    return im1.nx == im2.nx and im1.ny == im2.ny and im1.nz == im2.nz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def isImageEqual(im1, im2):
    """
    Checks if two images are equal (dimension, spacing, origin, variables).

    Parameters
    ----------
    im1 : :class:`Img`
        first image

    im2 : :class:`Img`
        second image

    Returns
    -------
    bool
        - `True` if the images are equal (same grid, same variable values, \
        variable names not checked)
        - `False` otherwise
    """
    # fname = 'isImageEqual'

    b = isImageDimensionEqual(im1, im2)
    if b:
        b = im1.sx == im2.sx and im1.sy == im2.sy and im1.sz == im2.sz
    if b:
        b = im1.ox == im2.ox and im1.oy == im2.oy and im1.oz == im2.oz
    if b:
        ind_isnan1 = np.isnan(im1.val)
        ind_isnan2 = np.isnan(im2.val)
        b = np.all(ind_isnan1 == ind_isnan2)
        if b:
            b = np.all(im1.val[~ind_isnan1] == im2.val[~ind_isnan2])
    return b
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def isPointSetEqual(ps1, ps2):
    """
    Checks if two point sets are equal (nb of points, nb of variables, variable values).

    Parameters
    ----------
    ps1 : :class:`PointSet`
        first point set

    ps2 : :class:`PointSet`
        second point set

    Returns
    -------
    bool
        - `True` if the point sets are equal (same number of points, same number \
        of variables, same variable values, variable names not checked)
        - `False` otherwise
    """
    # fname = 'isPointSetEqual'

    b = ps1.npt == ps2.npt and ps1.nv == ps2.nv
    if b:
        ind_isnan1 = np.isnan(ps1.val)
        ind_isnan2 = np.isnan(ps2.val)
        b = np.all(ind_isnan1 == ind_isnan2)
        if b:
            b = np.all(ps1.val[~ind_isnan1] == ps2.val[~ind_isnan2])
    return b
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def indicatorImage(im, ind=0, categ=None, return_categ=False, logger=None):
    """
    Retrieves the image of the indicator of each given category for the given variable.

    Parameters
    ----------
    im : :class:`Img`
        input image

    ind : int, default: 0
        index of the variable for which the indicator of categories are retrieved
        (negative integer for indexing from the end)

    categ : 1D array-like, optional
        list of category values for which the indicator are retrieved;
        by default (`None`): the list of all distinct values (in increasing
        order) taken by the variable of index `ind` in the input image is
        considered

    return_categ : bool
        indicates if the list of category values for which the indicator is
        retrieved (corresponding to `categ`) is returned or not

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        output image with indicator variable(s) (as many variable(s) as number
        of category values given by `categ`)

    categ : 1D array, optional
        category values for which the indicator variable is retrieved;
        returned if `return_categ=True`
    """
    fname = 'indicatorImage'

    # Check (set) ind
    if ind < 0:
        ind = im.nv + ind

    if ind < 0 or ind >= im.nv:
        err_msg = f'{fname}: invalid index'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Set categ if not given (None)
    if categ is None:
        categ = im.get_unique_one_var(ind=ind, logger=logger)

    ncateg = len(categ)

    # Initialize an image with ncateg variables
    im_out = Img(
        nx=im.nx, ny=im.ny, nz=im.nz,
        sx=im.sx, sy=im.sy, sz=im.sz,
        ox=im.ox, oy=im.oy, oz=im.oz,
        nv=ncateg, varname=[f'{im.varname[ind]}_ind{i:03d}' for i in range(ncateg)],
        logger=logger)

    # Compute each indicator variable
    for i, v in enumerate(categ):
        val = 1.*(im.val[ind]==v)
        val[np.where(np.isnan(im.val[ind]))] = np.nan
        im_out.val[i,...] = val

    if return_categ:
        out = im_out, np.asarray(categ)
    else:
        out = im_out

    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def gatherImages(
        im_list,
        varInd=None,
        keep_varname=False,
        rem_var_from_source=False,
        treat_image_one_by_one=False,
        logger=None):
    """
    Gathers images into one image.

    Parameters
    ----------
    im_list : 1D array-like of :class:`Img`
        images to be gathered, they should have the same grid dimensions (number
        of cell along each axis)

    varInd : 1D array-like or int, optional
        index(es) of the variables of each image from `im_list` to be retrieved
        (stored in the output image); by default (`None`): all variables are
        retrieved

    keep_varname : bool, default: False
        - if `True`: name of the variables are kept from the source images in \
        `im_list`
        - if `False`: default variable names are set

    rem_var_from_source : bool, default: False
        indicates if gathered variables are removed from the source images in
        `im_list` (this allows to save memory)

    treat_image_one_by_one : bool, default: False
        used only if `rem_var_from_source=True` (otherwise,
        `treat_image_one_by_one` is ignored (as it was set to `False`), because
        there is no need to deal with images one by one;

        - if `True`: images in `im_list` are treated one by one, i.e. the variables \
        to be gathered in each image are inserted in the output image and removed \
        from the source (slower, may save memory)
        - if `False`: all images in `im_list` are treated at once, i.e. variables \
        to be gathered from all images are inserted in the output image at once \
        (faster)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        output image containing variables to be gathered from all images in
        `im_list`; the order of variables is set as follows: variables of
        index(es) in `varInd` (unique index and in increasing order) of the image
        `im_list[0]`, then those of image `im_list[1]`, etc.
    """
    fname = 'gatherImages'

    if len(im_list) == 0:
        return None

    for i in range(1,len(im_list)):
        if not isImageDimensionEqual(im_list[0], im_list[i]):
            err_msg = f'{fname}: grid dimensions differ'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if np.sum([iv in range(im.nv) for im in im_list for iv in varInd]) != len(im_list)*len(varInd):
            err_msg = f'{fname}: invalid index-es'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    varname = None # default
    if keep_varname:
        if varInd is not None:
            varname = [im.varname[iv] for im in im_list for iv in varInd]
        else:
            varname = [im.varname[iv] for im in im_list for iv in range(im.nv)]

    if rem_var_from_source:
        # remove variable from source
        if treat_image_one_by_one:
            # treat images one by one
            val = np.empty(shape=(0, im_list[0].nz, im_list[0].ny, im_list[0].nx))
            if varInd is not None:
                ind = np.sort(np.unique(varInd))[::-1] # unique index in decreasing order (for removing variable...)
                for im in im_list:
                    val = np.concatenate((val, im.val[varInd]), 0)
                    for iv in ind:
                        im.remove_var(iv, logger=logger)
            else:
                for im in im_list:
                    val = np.concatenate((val, im.val), 0)
                    im.remove_allvar()
        else:
            # treat all images at once
            if varInd is not None:
                val = np.concatenate([im.val[varInd] for im in im_list], 0)
                ind = np.sort(np.unique(varInd))[::-1] # unique index in decreasing order (for removing variable...)
                for im in im_list:
                    for iv in ind:
                        im.remove_var(iv, logger=logger)
            else:
                val = np.concatenate([im.val for im in im_list], 0)
                for im in im_list:
                    im.remove_allvar()
    else:
        # not remove variable from source
        # ignore treat_image_one_by_one (as it was False)
        # treat_image_one_by_one = False # changed if needed: no need to treat images one by one...
        #
        # treat all images at once
        if varInd is not None:
            val = np.concatenate([im.val[varInd] for im in im_list], 0)
        else:
            val = np.concatenate([im.val for im in im_list], 0)

    im_out = Img(
            nx=im_list[0].nx, ny=im_list[0].ny, nz=im_list[0].nz,
            sx=im_list[0].sx, sy=im_list[0].sy, sz=im_list[0].sz,
            ox=im_list[0].ox, oy=im_list[0].oy, oz=im_list[0].oz,
            nv=val.shape[0], val=val, varname=varname,
            logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageContStat(im, op='mean', logger=None, **kwargs):
    """
    Computes "pixel-wise" statistics over all variables in an image.

    Parameters
    ----------
    im : :class:`Img`
        input image

    op : str {'min', 'max', 'mean', 'std', 'var', 'quantile'}, default: 'mean'
        statistic operator referring to the function `numpy.<op>`;
        note: `op='quantile'` requires the parameter
        `q=<sequence_of_quantile_to_compute>` that should be passed via `kwargs`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to `numpy.<op>` function, e.g.
        `ddof=1` if `op='std'` or`op='var'`

    Returns
    -------
    im_out : :class:`Img`
        image with same grid as the input image and one variable, the pixel-wise
        statistics according to operation `op` over all variables in the input
        image
    """
    fname = 'imageContStat'

    # Prepare operation
    if op == 'max':
        func = np.nanmax
        varname = [op]
    elif op == 'mean':
        func = np.nanmean
        varname = [op]
    elif op == 'min':
        func = np.nanmin
        varname = [op]
    elif op == 'std':
        func = np.nanstd
        varname = [op]
    elif op == 'var':
        func = np.nanvar
        varname = [op]
    elif op == 'quantile':
        func = np.nanquantile
        if 'q' not in kwargs:
            err_msg = f"({fname}): keyword argument 'q' required by `op='quantile'`"
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        varname = [op + '_' + str(v) for v in kwargs['q']]

    else:
        err_msg = f"({fname}): unkown operation '{op}'"
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                 sx=im.sx, sy=im.sy, sz=im.sz,
                 ox=im.ox, oy=im.oy, oz=im.oz,
                 nv=0, val=0.0,
                 logger=logger)

    vv = func(im.val.reshape(im.nv,-1), axis=0, **kwargs)
    vv = vv.reshape(-1, im.nxyz())
    for v, name in zip(vv, varname):
        im_out.append_var(v, varname=name, logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageListContStat(im_list, ind=0, op='mean', logger=None, **kwargs):
    """
    Computes "pixel-wise" statistics for one variable over all images in a list.

    Parameters
    ----------
    im_list : 1D array-like of :class:`Img`
        list of input images, they should have the same grid dimensions (number
        of cell along each axis)

    ind : int, default: 0
        index of the variable in each image from `im_list` to be considered

    op : str {'min', 'max', 'mean', 'std', 'var', 'quantile'}, default: 'mean'
        statistic operator referring to the function `numpy.<op>`;
        note: `op='quantile'` requires the parameter
        `q=<sequence_of_quantile_to_compute>` that should be passed via `kwargs`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to `numpy.<op>` function, e.g.
        `ddof=1` if `op='std'` or`op='var'`

    Returns
    -------
    im_out : :class:`Img`
        image with same grid as the input images and one variable, the pixel-wise
        statistics according to operation `op` over the variable of index `ind`
        in all images in `im_list`
    """
    fname = 'imageListContStat'

    # Check input images
    if not isinstance(im_list, list) and not (isinstance(im_list, np.ndarray) and im_list.ndim==1):
        err_msg = f'{fname}: first argument must be a list (or a 1d-array) of images'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if len(im_list) == 0:
        return None

    im0 = im_list[0]
    for im in im_list[1:]:
        if im.val.shape != im0.val.shape:
            err_msg = f'{fname}: images in list of incompatible size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    # Check (set) ind
    if ind < 0:
        ind = im0.nv + ind

    if ind < 0 or ind >= im0.nv:
        err_msg = f'{fname}: invalid index'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Prepare operation
    if op == 'max':
        func = np.nanmax
        varname = [op]
    elif op == 'mean':
        func = np.nanmean
        varname = [op]
    elif op == 'min':
        func = np.nanmin
        varname = [op]
    elif op == 'std':
        func = np.nanstd
        varname = [op]
    elif op == 'var':
        func = np.nanvar
        varname = [op]
    elif op == 'quantile':
        func = np.nanquantile
        if 'q' not in kwargs:
            err_msg = f"({fname}): keyword argument 'q' required by `op='quantile'`"
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        varname = [op + '_' + str(v) for v in kwargs['q']]

    else:
        err_msg = f"({fname}): unkown operation '{op}'"
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    im_out = Img(nx=im0.nx, ny=im0.ny, nz=im0.nz,
                 sx=im0.sx, sy=im0.sy, sz=im0.sz,
                 ox=im0.ox, oy=im0.oy, oz=im0.oz,
                 nv=0, val=0.0,
                 logger=logger)

    vv = func(np.asarray([im.val[ind] for im in im_list]).reshape(len(im_list),-1), axis=0, **kwargs)
    vv = vv.reshape(-1, im.nxyz())
    for v, name in zip(vv, varname):
        im_out.append_var(v, varname=name, logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageCategProp(im, categ, logger=None):
    """
    Computes "pixel-wise" proportions of given categories over all variables in an image.

    Parameters
    ----------
    im : :class:`Img`
        input image

    categ : 1D array-like
        list of category values for which the proportions are calculated

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        image with same grid as the input image and one variable per category
        value in `categ`: the pixel-wise proportions, over all variables
        in the input image
    """
    # fname = 'imageCategProp'

    # Array of categories
    categ_arr = np.array(categ, dtype=float).reshape(-1)

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=0, val=0.0,
                logger=logger)

    for i, code in enumerate(categ_arr):
        x = 1.0*(im.val.reshape(im.nv,-1) == code)
        np.putmask(x, np.isnan(im.val.reshape(im.nv,-1)), np.nan)
        imOut.append_var(np.mean(x, axis=0), varname=f'prop{i}', logger=logger)

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageListCategProp(im_list, categ, ind=0, logger=None):
    """
    Computes "pixel-wise" proportions of given categories for one variable over all images in a list.

    Parameters
    ----------
    im_list : 1D array-like of :class:`Img`
        list of input images, they should have the same grid dimensions (number
        of cell along each axis)

    categ : 1D array-like
        list of category values for which the proportions are calculated

    ind : int, default: 0
        index of the variable in each image from `im_list` to be considered

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        image with same grid as the input image and one variable per category
        value in `categ`: the pixel-wise proportions, over the variable of index
        `ind` in all images in `im_list`
    """
    fname = 'imageListCategProp'

    # Check input images
    if not isinstance(im_list, list) and not (isinstance(im_list, np.ndarray) and im_list.ndim==1):
        err_msg = f'{fname}: first argument must be a list (or a 1d-array) of images'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if len(im_list) == 0:
        return None

    im0 = im_list[0]
    for im in im_list[1:]:
        if im.val.shape != im0.val.shape:
            err_msg = f'{fname}: images in list of incompatible size'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    # Check (set) ind
    if ind < 0:
        ind = im0.nv + ind

    if ind < 0 or ind >= im0.nv:
        err_msg = f'{fname}: invalid index'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Array of categories
    categ_arr = np.array(categ, dtype=float).reshape(-1)

    im_out = Img(nx=im0.nx, ny=im0.ny, nz=im0.nz,
                 sx=im0.sx, sy=im0.sy, sz=im0.sz,
                 ox=im0.ox, oy=im0.oy, oz=im0.oz,
                 nv=0, val=0.0,
                 logger=logger)

    v = np.asarray([im.val[ind] for im in im_list]).reshape(len(im_list),-1)
    for i, code in enumerate(categ_arr):
        x = 1.0*(v == code)
        np.putmask(x, np.isnan(v), np.nan)
        im_out.append_var(np.mean(x, axis=0), varname=f'prop{i}', logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageEntropy(im, varInd=None, varIndList=None, logger=None):
    """
    Computes "pixel-wise" entropy from proportions given as variables in an image.

    For each grid cell of (single) index i, the entropy is defined as

    .. math::
        H[i] = - \\sum_{v} v[i] \\cdot \\log_{n}(v[i])

    where :math:`v` loops on each considered variable, and :math:`n` is the number
    of considered variables, assuming that the variables are proportions that sum
    to 1.0 in each grid cell, i.e. :math:`\\sum_{v} v[i]` should be equal to 1.0,
    for any i.

    Parameters
    ----------
    im : :class:`Img`
        input image

    varInd : 1D array-like, optional
        indexes of the variables of the input image to be taken into account;
        by default (`None`): all variables are considered

    varIndList : int or 1D array-like of ints, or None (default)
        deprecated (used in place of `varInd` if `varInd=None`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        image with same grid as the input image and one variable, the pixel-wise
        entropy (see above)
    """
    fname = 'imageEntropy'

    if varInd is None:
        varInd = varIndList

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        # Check if each index is valid
        if np.sum([iv in range(im.nv) for iv in varInd]) != len(varInd):
            err_msg = f'{fname}: invalid index-es'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        varInd = range(im.nv)

    if len(varInd) < 2:
        err_msg = f'{fname}: at least 2 indexes should be given'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                 sx=im.sx, sy=im.sy, sz=im.sz,
                 ox=im.ox, oy=im.oy, oz=im.oz,
                 nv=1, val=np.nan,
                 name=im.name,
                 logger=logger)

    t = 1. / np.log(len(varInd))

    for iz in range(im.nz):
        for iy in range(im.ny):
            for ix in range(im.nx):
                s = 0
                e = 0
                ok = True
                for iv in varInd:
                    p = im.val[iv][iz][iy][ix]
                    if np.isnan(p) or p < 0:
                        ok = False
                        break
                    s = s + p
                    if p > 1.e-10:
                        e = e - p*np.log(p)

                if ok and abs(s-1.0) > 1.e-5:
                    ok = False
                if ok:
                    im_out.val[0][iz][iy][ix] = t*e

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageCategFromImageOfProp(
        im,
        mode='most_probable',
        target_prop=None,
        varInd=None,
        categ=None,
        logger=None):
    """
    Retrieves a categorical image from proportions given as variables in an image.

    For each grid cell, the output category is defined according to the values of
    the considered variables from the input image interpreted as proportions, and
    according to the specified mode: target proportions over all grid cells, or
    most probable (frequent) category in each grid cell.

    Parameters
    ----------
    im : :class:`Img`
        input image

    mode : str {'most_probable', 'target_prop'}, default: 'most_probable'
        defines how is computed the output variable:

        - 'most_probable': most probable category (index) in each grid cell
        - 'target_prop': category (index) such that the proportions over all \
        image grid cells match as much as possible the proportions given by \
        `target_prop`

    target_prop : 1D array-like, optional
        target proportions of categories (indexes), used if `mode='target_prop'`;
        by default (`None`): the target proportions are set according to the
        proportions in the input image, i.e. `target_prop[i] = mean(im.val[varInd[i]])`
        where `varInd` are the indexes of the considered variables in `im`

    varInd : 1D array-like, optional
        indexes of the variables of the input image to be taken into account;
        by default (`None`): all variables are considered

    categ : 1D array-like, optional
        category values to be assigned in place of the category indexes in the
        output image, i.e. output index i (corresponding to variable `varInd[i]`
        in the input image) is replaced by `categ[i]` (note that the length of
        `categ` must be the same as the length of `varInd`);
        by default (`None`): `categ[i]=i` is used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_out : :class:`Img`
        image with one variable, a category index (or value) according to the
        used mode
    """
    fname = 'imageCategFromImageOfProp'

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        # Check if each index is valid
        if np.sum([iv in range(im.nv) for iv in varInd]) != len(varInd):
            err_msg = f'{fname}: invalid index-es'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        varInd = range(im.nv)

    n = len(varInd)

    if n < 2:
        err_msg = f'{fname}: at least 2 indexes should be given'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Array of categories
    if categ is not None:
        categ_arr = np.array(categ, dtype=float).reshape(-1)
        if len(categ_arr) != n:
            err_msg = f'{fname}: `categ` of incompatible length'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        categ_arr = np.arange(float(n))

    val = im.val[varInd,:,:,:] # (copy)

    if mode == 'most_probable':
        # Get index (id) of the greatest proportion (at each cell)
        id = np.argsort(-val, axis=0)[0]
        v = np.asarray([categ_arr[i] for i in id.ravel()]).reshape(id.shape)
        np.putmask(v, np.any(np.isnan(val), axis=0), np.nan)

    elif mode == 'target_prop':
        # Array of target proportions
        if target_prop is not None:
            target_prop_arr = np.array(target_prop, dtype=float).reshape(-1)
            if len(target_prop_arr) != n:
                err_msg = f'{fname}: `target_prop` of incompatible length'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        else:
            target_prop_arr = np.maximum(0.0, np.minimum(1.0, np.nanmean(val, axis=(1, 2, 3))))

        # Check target proportions
        if np.any((target_prop_arr < 0.0, target_prop_arr > 1.0)):
            err_msg = f'{fname}: `target_prop` invalid (value not in [0,1])'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        if not np.isclose(target_prop_arr.sum(), 1.0):
            err_msg = f'{fname}: `target_prop` invalid (do not sum to 1.0)'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        # Fill the output variable by starting with the index of the smallest target proportion
        id_prop = np.argsort(target_prop_arr)
        # Initialization
        cells = np.any(np.isnan(val), axis=0)   # grid image cell already set
        id = np.zeros_like(cells, dtype='int')  # output id on grid image
        # Treat all variable indexes
        for i in range(n):
            j = id_prop[i]
            a = val[j]
            q = np.quantile(a[~cells], 1.0-target_prop_arr[j])
            cells_ind = np.all((a > q, ~cells), axis=0)
            id[cells_ind] = j
            cells = np.any((cells, cells_ind), axis=0)

        # Treat remaining cells (not yet assigned), using most probable index
        id[~cells] = np.argsort(-val[:,~cells], axis=0)[0]
        v = np.asarray([categ_arr[i] for i in id.ravel()]).reshape(id.shape)
        np.putmask(v, np.any(np.isnan(val), axis=0), np.nan)

    im_out = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                 sx=im.sx, sy=im.sy, sz=im.sz,
                 ox=im.ox, oy=im.oy, oz=im.oz,
                 nv=1, val=v,
                 name=mode,
                 logger=logger)

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def interpolateImage(
        im,
        categVar=None,
        nx=None, ny=None, nz=None,
        sx=None, sy=None, sz=None,
        ox=None, oy=None, oz=None,
        logger=None,
        **kwargs):
    """
    Interpolates (each variable of) an image on a given grid, and returns an output image.

    The output image grid geometry is defined as follows for the x axis (similar
    for y and z axes):

    - if `ox` is None:
        - `ox = im.ox` is used

    - if `nx` is None and `sx` is not None:
        - `nx = int(np.round(im.nx*im.sx/sx))` is used

    - if `nx` is not None and `sx` is None:
        - `sx = im.nx*im.sx/nx` is used

    - if `nx` is None and `sx` is None:
        - `nx = im.nx` and `sx = im.sx` are used

    Note: this function allows for example to refine an image.

    Parameters
    ----------
    im : :class:`Img`
        input image

    categVar : 1D array-like of bools, optional
        sequence of `im.nv`:

        - `categVar[i]=True` : the variable i is treated as a categorical variable
        - `categVar[i]=False`: the variable i is treated as a continuous variable \
        by default (`None`): all variables are treated as continuous variable

    nx : int, optional
        number of grid cells along x axis in the output image;
        see above for possible inputs

    ny : int, optional
        number of grid cells along y axis in the output image;
        see above for possible inputs

    nz : int, optional
        number of grid cells along z axis in the output image;
        see above for possible inputs

    sx : float, optional
        cell size along x axis in the output image;
        see above for possible inputs

    sy : float, optional
        cell size along y axis in the output image;
        see above for possible inputs

    sz : float, optional
        cell size along z axis in the output image;
        see above for possible inputs

    ox : float, optional
        origin of the grid along x axis (x coordinate of cell border) in the output image;
        see above for possible inputs

    oy : float, optional
        origin of the grid along y axis (y coordinate of cell border) in the output image;
        see above for possible inputs

    oz : float, optional
        origin of the grid along z axis (z coordinate of cell border) in the output image;
        see above for possible inputs

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the interpolator (:class:`Img_interp_func`),
        e.g. keys items 'order', 'mode', 'cval'

    Returns
    -------
    im_out : :class:`Img`
        image with all variables of the input image, interpolated on the
        specified grid
    """
    fname = 'interpolateImage'

    # Variable type
    if categVar is None:
        categVar = np.zeros(im.nv, dtype='bool')
    else:
        categVar = np.atleast_1d(categVar)

    if len(categVar) != im.nv:
        err_msg = f'{fname}: `categVar` of incompatible length'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Set output image grid
    if ox is None:
        ox = im.ox

    if oy is None:
        oy = im.oy

    if oz is None:
        oz = im.oz

    if nx is None:
        if sx is None:
            nx = im.nx
            sx = im.sx
        else:
            nx = int(np.round(im.nx*im.sx / sx))
    elif sx is None:
            sx = im.nx*im.sx / nx

    if ny is None:
        if sy is None:
            ny = im.ny
            sy = im.sy
        else:
            ny = int(np.round(im.ny*im.sy / sy))
    elif sy is None:
            sy = im.ny*im.sy / ny

    if nz is None:
        if sz is None:
            nz = im.nz
            sz = im.sz
        else:
            nz = int(np.round(im.nz*im.sz / sz))
    elif sz is None:
            sz = im.nz*im.sz / nz

    # Initialize output image
    im_out = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=im.nv, val=np.nan, varname=im.varname,
                 logger=logger)

    # Points where the variables will be evaluated by interpolation
    points = np.array((im_out.xx().reshape(-1), im_out.yy().reshape(-1), im_out.zz().reshape(-1))).T
    for i in range(im.nv):
        if categVar[i]:
            # Get the image of indicator variables
            im_indic, categVal = indicatorImage(im, ind=i, return_categ=True, logger=logger)
            # Get the interpolator function for each indicator variable
            interp_indic = [Img_interp_func(im_indic, ind=j, logger=logger, **kwargs) for j in range(len(categVal))]
            # Interpolate each indicator variable at points (above)
            v = [interp(points) for interp in interp_indic]

            # Define image of indicator variables on the output image grid (reuse im_indic)
            im_indic = Img(nx=nx, ny=ny, nz=nz,
                           sx=sx, sy=sy, sz=sz,
                           ox=ox, oy=oy, oz=oz,
                           nv=len(categVal), val=v,
                           logger=logger)

            # Get the image of the resulting categorical variable from the indicator variables
            im_tmp = imageCategFromImageOfProp(im_indic, mode='target_prop', categ=categVal)
            im_out.val[i,:,:,:] = im_tmp.val[0]
        else:
            # Interpolate the variable at points (above)
            v = Img_interp_func(im, ind=i, logger=logger, **kwargs)(points)
            im_out.val[i,:,:,:] = v.reshape(im_out.val.shape[1:])

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sampleFromPointSet(ps, size, mask_val=None, seed=None, logger=None):
    """
    Samples random points from PointSet object and return a point set.

    Parameters
    ----------
    ps : :class:`PointSet`
        point set to sample from

    size : int
        number of points to be sampled

    mask_val : 1D array-like, optional
        sequence of length `ps.npt`, indicating for each point if it can be
        sampled (value not equal to 0) or not (otherwise)

    seed : int, optional
        seed for initializing random number generator

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps_out : :class:`PointSet`
        point set containing the sample points
    """
    fname = 'sampleFromPointSet'

    if seed is not None:
        np.random.seed(seed)

    if mask_val is not None:
        mask_val = np.asarray(mask_val).reshape(-1)
        if mask_val.size != ps.npt:
            err_msg = f'{fname}: size of `mask_val` invalid'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        indexes = np.where(mask_val != 0)[0]
        if size > len(indexes):
            err_msg = f'{fname}: `size` greater than number of active points in `ps`'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        indexes = ps.npt
        if size > indexes:
            err_msg = f'{fname}: `size` greater than number of points in `ps`'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    sample_indexes = np.sort(np.random.choice(indexes, size, replace=False))

    # Return the new object
    return PointSet(
            npt=size,
            nv=ps.nv,
            val=ps.val[:, sample_indexes],
            varname=ps.varname,
            name='sample_from_' + ps.name)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sampleFromImage(im, size, mask_val=None, seed=None, logger=None):
    """
    Samples random points from Img object and returns a point set.

    Coordinates of the sample points correspond to the center of the grid cell
    centers.

    Parameters
    ----------
    im : :class:`Img`
        image to sample from

    size : int
        number of points to be sampled

    mask_val : 1D array-like, optional
        sequence of length `im.nxyz()`, indicating for each grid cell if it can
        be sampled (value not equal to 0) or not (otherwise)

    seed : int, optional
        seed for initializing random number generator

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps_out : :class:`PointSet`
        point set containing the sample points
    """
    fname = 'sampleFromImage'

    if seed is not None:
        np.random.seed(seed)

    if mask_val is not None:
        mask_val = np.asarray(mask_val).reshape(-1)
        if mask_val.size != im.nxyz():
            err_msg = f'{fname}: size of `mask_val` invalid'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        indexes = np.where(mask_val != 0)[0]
        if size > len(indexes):
            err_msg = f'{fname}: `size` greater than number of active grid cells in `im`'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    else:
        indexes = im.nxyz()
        if size > indexes:
            err_msg = f'{fname}: `size` greater than number of grid cells in `im`'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    sample_indexes = np.sort(np.random.choice(indexes, size, replace=False))

    x = im.xx().reshape(-1)[sample_indexes]
    y = im.yy().reshape(-1)[sample_indexes]
    z = im.zz().reshape(-1)[sample_indexes]
    val = im.val.reshape(im.nv, -1)[:, sample_indexes]

    # Return the sampled point set
    return PointSet(
            npt=size,
            nv=3+im.nv,
            val=np.vstack((x, y, z, val)),
            varname=np.hstack((['x', 'y', 'z'], im.varname)),
            name='sample_from_' + im.name)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def extractRandomPointFromImage(im, npt, seed=None, logger=None):
    """
    Extracts random points from an image (at center of grid cells) and return
    the corresponding point set.

    Deprecated, use function `sampleFromImage`.

    Parameters
    ----------
    im : :class:`Img`
        image to sample from

    npt : int
        number of points to be sampled (if greater than the number of image grid
        cells, every cell is sampled)

    seed : int, optional
        seed for initializing random number generator

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps_out : :class:`PointSet`
        point set containing the sample points
    """
    fname = 'extractRandomPointFromImage'

    if npt <= 0:
        err_msg = f'{fname}: number of points negative or zero (`npt={npt}`)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if npt >= im.nxyz():
        return imageToPointSet(im)

    if seed is not None:
        np.random.seed(seed)

    # Get random single grid indexes
    ind_grid = np.random.choice(np.arange(im.nxyz()), size=npt, replace=False)

    # Get grid indiexes along each axis
    ind_ixyz = np.array([singleGridIndexToGridIndex(i, im.nx, im.ny, im.nz) for i in ind_grid])

    # Get points coordinates
    x = im.ox + (ind_ixyz[:,0]+0.5)*im.sx
    y = im.oy + (ind_ixyz[:,1]+0.5)*im.sy
    z = im.oz + (ind_ixyz[:,2]+0.5)*im.sz

    # Get value of every variable at points
    v = np.array([im.val.reshape(im.nv,-1)[:,i] for i in ind_grid])

    # Initialize point set
    ps = PointSet(npt=npt, nv=3+im.nv, val=0.0)

    # Set points coordinates
    ps.set_var(val=x, varname='X', ind=0, logger=logger)
    ps.set_var(val=y, varname='Y', ind=1, logger=logger)
    ps.set_var(val=z, varname='Z', ind=2, logger=logger)

    # Set next variable(s)
    for i in range(im.nv):
        ps.set_var(val=v[:,i], varname=im.varname[i], ind=3+i, logger=logger)

    return ps
# ----------------------------------------------------------------------------

# === Read / Write function below ===

# ----------------------------------------------------------------------------
def readVarsTxt(
        fname,
        missing_value=None,
        delimiter=' ',
        comments='#',
        usecols=None,
        logger=None):
    """
    Reads variables (data table) from a txt file.

    The file is in the following format::

        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (str) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    fname : str or file handle
        name of the file or file handle

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line;
        note: "empty field" after splitting is ignored, then if white space is
        used as delimiter (default), one or multiple white spaces can be used as
        the same delimiter

    comments : str, default:'#'
        lines starting with that string are treated as comments

    usecols : 1D array-like or int, optional
        column index(es) to be read (first column corresponds to index 0);
        by default, all columns are read

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    varname : list
        list of variable names, `varname[i]` is the name of the variable of
        index i

    val : 2D array
        values of the variables, with `val[:, i]` the values of variable of
        index i
    """
    funcname = 'readVarsTxt'

    # Check comments identifier
    if comments is not None and comments == '':
        err_msg = f'{funcname}: `comments` cannot be an empty string, use `comments=None` to disable comments'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Use pandas.read_csv to read (the rest of) the file (variable names and variable values)
    #    much faster than numpy.loadtxt
    if usecols is not None and isinstance(usecols, int):
        usecols = (usecols,)

    data_table = pd.read_csv(fname, comment=comments, delimiter=delimiter, usecols=usecols)
    varname = data_table.columns.to_list()
    val = data_table.values.astype('float')

    # # --- With numpy.loadtxt - start ...
    # if isinstance(fname, str):
    #     ff = open(fname, 'r')
    # else:
    #     ff = fname
    #
    # # Skip header (commented lines) and read the first line (after header)
    # line = ff.readline()
    # if comments is not None:
    #     nc = len(comments)
    #     while line[:nc] == comments:
    #         line = ff.readline()
    #
    # # Set variable names
    # varname = [s for s in line.replace('\n','').replace('\r','').split(delimiter) if s != ''] # keep only non empty field
    # if usecols is not None:
    #     if isinstance(usecols, int):
    #         usecols = (usecols,)
    #     varname = [varname[i] for i in usecols]
    #
    # # Read the rest of the file
    # val = np.loadtxt(ff, ndmin=2, delimiter=delimiter, comments=comments, usecols=usecols)
    #
    # if isinstance(fname, str):
    #     ff.close()
    # # --- With numpy.loadtxt - end ...

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(val, val == missing_value, np.nan)

    return (varname, val)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeVarsTxt(
        fname,
        varname,
        val,
        missing_value=None,
        delimiter=' ',
        usecols=None,
        fmt="%.10g",
        logger=None):
    """
    Writes variables (data table) in a txt file.

    The file is in the following format::

        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (str) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    fname : str or file handle
        name of the file or file handle

    varname : 1D array-like of strs
        sequence of variable names, `varname[i]` is the name of the variable of
        index i

    val : 2D array
        values of the variables, with `val[:,i]` the values of variable of
        index i

    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line

    usecols : 1D array-like or int, optional
        column index(es) to be read (first column corresponds to index 0);
        by default, all columns are read

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    funcname = 'writeVarsTxt'

    varname = np.asarray(varname).reshape(-1)
    # if not isinstance(varname, list):
    #     err_msg = f'{fname}: `varname` invalid, should be a list'
    #     if logger: logger.error(err_msg)
    #     raise ImgError(err_msg)

    if val.ndim != 2 or val.shape[1] != len(varname):
        err_msg = f'{funcname}: `val` and `varname` are incompatible'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Extract columns if needed
    if usecols is not None and isinstance(usecols, int):
        usecols = (usecols,)

    if usecols is not None:
        varname = [varname[i] for i in usecols]
        val = val[:, usecols]

    # Set header (variable names)
    header = delimiter.join(varname)

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(val, np.isnan(val), missing_value)

    np.savetxt(fname, val, comments='', header=header, delimiter=delimiter, fmt=fmt)

    # Replace missing_value by np.nan (restore)
    if usecols is None and missing_value is not None:
        # if usecols is None, val is not a copy!
        np.putmask(val, val == missing_value, np.nan)

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readGridInfoFromHeaderTxt(
        filename,
        nx=1, ny=1, nz=1,
        sx=1.0, sy=1.0, sz=1.0,
        ox=0.0, oy=0.0, oz=0.0,
        sorting='+X+Y+Z',
        header_str='#',
        max_lines=None,
        key_nx=['nx', 'nxcell', 'nrow', 'nrows'],
        key_ny=['ny', 'nycell', 'ncol', 'ncols'],
        key_nz=['nz', 'nzcell', 'nlay', 'nlays'],
        key_sx=['sx', 'xsize', 'xcellsize'],
        key_sy=['sy', 'ysize', 'ycellsize'],
        key_sz=['sz', 'zsize', 'zcellsize'],
        key_ox=['ox', 'xorigin', 'xllcorner'],
        key_oy=['oy', 'yorigin', 'yllcorner'],
        key_oz=['oz', 'zorigin', 'zllcorner'],
        key_sorting=['sorting'],
        get_sorting=False,
        logger=None):
    """
    Reads grid geometry information, and sorting mode of filling, from the header in a file.

    The grid geometry , i.e.

    - (nx, ny, nz), grid size, number of cells along each direction
    - (sx, sy, sz), grid cell size along each direction
    - (ox, oy, oz), grid origin, coordinates of the bottom-lower-left corner

    is retrieved from the header (lines starting with the `header_str` identifier
    in the beginning of the file). Default values are used if not specified.

    The sorting mode (for filling the grid with the variables) is also retrieved
    if asked for. If not specified, the default mode is `sorting='+X+Y+Z'`, which
    means that the grid is filled with

    - x index increases, then y index increases, then z index increases

    The string `sorting` should have 6 characters (see exception below):

    - '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'

    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with

    - first, `sorting[1]` index decreases (resp. increases) if `sorting[0]='-'` (resp. `sorting[0]='+'`)
    - then, `sorting[3]` index decreases (resp. increases) if `sorting[2]='-'` (resp. `sorting[2]='+'`)
    - then, `sorting[5]` index decreases (resp. increases) if `sorting[4]='-'` (resp. `sorting[4]='+'`)

    As an exception, if `nz=1`, the string `sorting` can have 4 characters:

    - '[+|-][X|Y][+|-][X|Y]'

    and it is then interpreted as above by appending '+Z'.

    Note that

    - the string `sorting` is case insensitive,
    - the validity of the string `sorting` is not checked in this function.

    Example of file::

        # [...]
        # # GRID - NUMBER OF CELLS
        # NX <int>
        # NY <int>
        # NZ <int>
        # # GRID - CELL SIZE
        # SX <float>
        # SY <float>
        # SZ <float>
        # # GRID - ORIGIN (bottom-lower-left corner)
        # OX <float>
        # OY <float>
        # OZ <float>
        # # GRID - FILLING
        # SORTING +X+Y+Z
        # [...]
        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (str) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Only the lines starting with the string `header_str` ('#' by default) in the
    beginning of the file are read, but at maximum `max_lines` lines (if None,
    not limited).

    Parameters
    ----------
    filename : str
        name of the file

    nx : int, default: 1
        number of grid cells along x axis used as default

    ny : int, default: 1
        number of grid cells along y axis used as default

    nz : int, default: 1
        number of grid cells along z axis used as default

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    sx : float, default: 1.0
        cell size along x axis used as default

    sy : float, default: 1.0
        cell size along y axis used as default

    sz : float, default: 1.0
        cell size along z axis used as default

        Note: `(sx, sy, sz)` is the cell size

    ox : float, default: 0.0
        origin of the grid along x axis (x coordinate of cell border)
        used as default

    oy : float, default: 0.0
        origin of the grid along y axis (y coordinate of cell border)
        used as default

    oz : float, default: 0.0
        origin of the grid along z axis (z coordinate of cell border)
        used as default

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    sorting : str, default: '+X+Y+Z'
        describes the way to fill the grid (see above)

    header_str : str, default: '#'
        only lines starting with `header_str` in the beginning of the file are
        treated, use `header_str=None` for no line identifier

    max_lines : int, optional
        maximum number of lines read (by default: unlimited); note: if
        `header_str=None` and `max_lines=None`, the whole file will be read

    key_nx : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `nx`

    key_ny : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `ny`

    key_nz : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `nz`

    key_sx : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `sx`

    key_sy : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `sy`

    key_sz : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `sz`

    key_ox : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `ox`

    key_oy : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `oy`

    key_oz : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `oz`

    key_sorting : 1D array-like of strs, or str
        possible key words (case insensitive) for entry `sorting`

    get_sorting : bool
        indicates if sorting mode is retrieved (`True`) or not (`False`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz)[, sorting]) : 3-tuple [or 4-tuple]
        - grid geometry, where
            - `(nx, ny, nz)` : (3-tuple of ints) number of cells along each axis,
            - `(sx, sy, sz)` : (3-tuple of floats) cell size along each axis
            - `(ox, oy, oz)` : (3-tuple of floats) coordinates of the origin \
            (bottom-lower-left corner)

        - if `get_sorting=True`:
            - `sorting` : (str) string of length 6 describing the sorting \
            mode of filling
    """
    fname = 'readGridInfoFromHeaderTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Check header_str identifier
    if header_str is not None:
        if header_str == '':
            err_msg = f'{fname}: `header_str` identifier cannot be an empty string, use `header_str=None` instead'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        else:
            nhs = len(header_str)

    # Flag to indicate if entry is read
    nx_flag, ny_flag, nz_flag = False, False, False
    sx_flag, sy_flag, sz_flag = False, False, False
    ox_flag, oy_flag, oz_flag = False, False, False
    sorting_flag = False

    # Read the file
    nline = 0
    # Open the file in read mode
    with open(filename,'r') as ff:
        while True: # break to exit
            if max_lines is not None and nline >= max_lines:
                break

            # Read next line
            line = ff.readline()
            if not line:
                # End of line reached
                break

            if header_str is not None:
                if line[:nhs] == header_str:
                    line = line[nhs:]
                else:
                    break

            # Treat current line
            line_s = line[nhs:].split()
            k = 0
            while k < len(line_s)-1:
                entry = line_s[k].lower()
                if entry in key_nx: # entry for nx ?
                    if nx_flag:
                        err_msg = f'{fname}: more than one entry for `nx`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        nx = int(line_s[k+1])
                        k = k+2
                        nx_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `nx`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_ny: # entry for ny ?
                    if ny_flag:
                        err_msg = f'{fname}: more than one entry for `ny`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        ny = int(line_s[k+1])
                        k = k+2
                        ny_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `ny`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_nz: # entry for nz ?
                    if nz_flag:
                        err_msg = f'{fname}: more than one entry for `nz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        nz = int(line_s[k+1])
                        k = k+2
                        nz_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `nz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_sx: # entry for sx ?
                    if sx_flag:
                        err_msg = f'{fname}: more than one entry for `sx`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        sx = float(line_s[k+1])
                        k = k+2
                        sx_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `sx`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_sy: # entry for sy ?
                    if sy_flag:
                        err_msg = f'{fname}: more than one entry for `sy`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        sy = float(line_s[k+1])
                        k = k+2
                        sy_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `sy`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_sz: # entry for sz ?
                    if sz_flag:
                        err_msg = f'{fname}: more than one entry for `sz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        sz = float(line_s[k+1])
                        k = k+2
                        sz_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `sz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_ox: # entry for ox ?
                    if ox_flag:
                        err_msg = f'{fname}: more than one entry for `ox`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        ox = float(line_s[k+1])
                        k = k+2
                        ox_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `ox`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_oy: # entry for oy ?
                    if oy_flag:
                        err_msg = f'{fname}: more than one entry for `oy`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        oy = float(line_s[k+1])
                        k = k+2
                        oy_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `oy`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_oz: # entry for oz ?
                    if oz_flag:
                        err_msg = f'{fname}: more than one entry for `oz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        oz = float(line_s[k+1])
                        k = k+2
                        oz_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `oz`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                elif entry in key_sorting and get_sorting: # entry for sorting (and get_sorting)?
                    if sorting_flag:
                        err_msg = f'{fname}: more than one entry for `sorting`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                    try:
                        sorting = line_s[k+1]
                        k = k+2
                        sorting_flag = True
                    except:
                        err_msg = f'{fname}: reading entry for `sorting`'
                        if logger: logger.error(err_msg)
                        raise ImgError(err_msg)

                else:
                    k = k+1

            # "end while"
            nline = nline+1

    if get_sorting:
        out = ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz), sorting)
    else:
        out = ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz))

    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageTxt(
        filename,
        nx=1, ny=1, nz=1,
        sx=1.0, sy=1.0, sz=1.0,
        ox=0.0, oy=0.0, oz=0.0,
        sorting='+X+Y+Z',
        missing_value=None,
        delimiter=' ',
        comments='#',
        usecols=None,
        logger=None):
    """
    Reads an image from a txt file, including grid geometry, and sorting mode of filling.

    The image grid geometry , i.e.

    - (nx, ny, nz), grid size, number of cells along each direction
    - (sx, sy, sz), grid cell size along each direction
    - (ox, oy, oz), grid origin, coordinates of the bottom-lower-left corner

    is retrieved from the header (lines starting with the `comments` identifier
    in the beginning of the file). Default values given by the parameters are
    used if information is not written in the file.

    The number n of values (see below) for each variable should be equal to
    `nx*ny*nz`. The grid is filled according to the specified `sorting` mode.
    By default `sorting='+X+Y+Z'`, which means that the grid is filled with

    - x index increases, then y index increases, then z index increases

    The string `sorting` must have 6 characters (see exception below):

    - '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'

    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with

    - first, `sorting[1]` index decreases (resp. increases) if `sorting[0]='-'` (resp. `sorting[0]='+'`)
    - then, `sorting[3]` index decreases (resp. increases) if `sorting[2]='-'` (resp. `sorting[2]='+'`)
    - then, `sorting[5]` index decreases (resp. increases) if `sorting[4]='-'` (resp. `sorting[4]='+'`)

    As an exception, if `nz=1`, the string `sorting` can have 4 characters:

    - '[+|-][X|Y][+|-][X|Y]'

    and it is then interpreted as above by appending '+Z'.

    Note that the string `sorting` is case insensitive.

    Grid geometry and sorting mode of filling is retrieved from the header of
    the file (if present), i.e. the commented lines in the beginning of the file
    (see also function :func:`readGridInfoFromHeaderTxt`).

    Example of file::

        # [...]
        # # GRID - NUMBER OF CELLS
        # NX <int>
        # NY <int>
        # NZ <int>
        # # GRID - CELL SIZE
        # SX <float>
        # SY <float>
        # SZ <float>
        # # GRID - ORIGIN (bottom-lower-left corner)
        # OX <float>
        # OY <float>
        # OZ <float>
        # # GRID - FILLING
        # SORTING +X+Y+Z
        # [...]
        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (str) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    filename : str
        name of the file

    nx : int, default: 1
        number of grid cells along x axis used as default

    ny : int, default: 1
        number of grid cells along y axis used as default

    nz : int, default: 1
        number of grid cells along z axis used as default

        Note: `(nx, ny, nz)` is the grid dimension (in number of cells)

    sx : float, default: 1.0
        cell size along x axis used as default

    sy : float, default: 1.0
        cell size along y axis used as default

    sz : float, default: 1.0
        cell size along z axis used as default

        Note: `(sx, sy, sz)` is the cell size

    ox : float, default: 0.0
        origin of the grid along x axis (x coordinate of cell border)
        used as default

    oy : float, default: 0.0
        origin of the grid along y axis (y coordinate of cell border)
        used as default

    oz : float, default: 0.0
        origin of the grid along z axis (z coordinate of cell border)
        used as default

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid

    sorting : str, default: '+X+Y+Z'
        describes the way to fill the grid (see above)

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line;
        note: "empty field" after splitting is ignored, then if white space is
        used as delimiter (default), one or multiple white spaces can be used as
        the same delimiter

    comments : str, default:'#'
        lines starting with that string are treated as comments, such lines in
        the beginning of the file constitute the header of the file from which
        the grid geometry and sorting mode (if written) are read

    usecols : 1D array-like or int, optional
        column index(es) to be read (first column corresponds to index 0);
        by default, all columns are read

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im : :class:`Img`
        image (read from the file)
    """
    fname = 'readImageTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Check comments identifier
    if comments is None or comments == '':
        err_msg = f'{fname}: `comments` cannot be an empty string (nor `None`)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Read grid geometry information and sorting mode from header
    try:
        ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz), sorting) = readGridInfoFromHeaderTxt(
                filename,
                nx=nx, ny=ny, nz=nz,
                sx=sx, sy=sy, sz=sz,
                ox=ox, oy=oy, oz=oz,
                sorting=sorting,
                header_str=comments,
                get_sorting=True,
                logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: grid geometry information cannot be read'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg) from exc

    # Deal with sorting
    if len(sorting) == 4:
        sorting = sorting + '+Z'

    if len(sorting) != 6:
        err_msg = f'{fname}: `sorting` (string) invalid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    sorting = sorting.lower() # tranform to lower case
    s = sorting[1] + sorting[3]+ sorting[5]
    if s == 'xyz':
        sha = (nz, ny, nx)
        tr = (1, 2, 3)
    elif s == 'xzy': # "transpose_xyz_to_xzy" to do
        sha = (ny, nz, nx)
        tr = (2, 1, 3)
    elif s == 'yxz': # "transpose_xyz_to_yxz" to do
        sha = (nz, nx, ny)
        tr = (1, 3, 2)
    elif s == 'yzx': # "transpose_xyz_to_yzx" to do
        sha = (nx, nz, ny)
        tr = (2, 3, 1)
    elif s == 'zxy': # "transpose_xyz_to_zxy" to do
        sha = (ny, nx, nz)
        tr = (3, 1, 2)
    elif s == 'zyx': # "transpose_xyz_to_zyx" to do
        sha = (nx, ny, nz)
        tr = (3, 2, 1)
    else:
        err_msg = f'{fname}: `sorting` (string) invalid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    flip = [1, 1, 1]
    for i in range(3):
        s = sorting[2*i]
        if s == '-':
            flip[i] = -1
        elif s != '+':
            err_msg = f'{fname}: `sorting` (string) invalid'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    # Read variale names and values from file
    try:
        varname, val = readVarsTxt(filename, missing_value=missing_value, delimiter=delimiter, comments=comments, usecols=usecols, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: variables names / values cannot be read'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg) from exc

    if val.shape[0] != nx*ny*nz:
        err_msg = f'{fname}: number of grid cells and number of values for each variable differ'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Reorganize val array according to sorting, final shape: (len(varname), nz, ny, nx)
    val = val.T.reshape(-1, *sha)[:, ::flip[2], ::flip[1], ::flip[0]].transpose(0, *tr)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, len(varname), val, varname, filename, logger=logger)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageTxt(
        filename,
        im,
        sorting='+X+Y+Z',
        missing_value=None,
        delimiter=' ',
        comments='#',
        endofline='\n',
        usevars=None,
        fmt="%.10g",
        logger=None):
    """
    Writes an image in a txt file, including grid geometry, and sorting mode of filling.

    The grid geometry and the sorting mode of filling is written in the beginning
    of the file with lines starting with the string `comments`.

    By default, `sorting='+X+Y+Z'` is used, which means that the
    grid is filled (with values as they are written) with

    - x index increases, then y index increases, then z index increases

    The string `sorting` should have 6 characters (see exception below):

    - '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'

    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with

    - first, `sorting[1]` index decreases (resp. increases) if `sorting[0]='-'` (resp. `sorting[0]='+'`)
    - then, `sorting[3]` index decreases (resp. increases) if `sorting[2]='-'` (resp. `sorting[2]='+'`)
    - then, `sorting[5]` index decreases (resp. increases) if `sorting[4]='-'` (resp. `sorting[4]='+'`)

    As an exception, if `nz=1`, the string `sorting` can have 4 characters:

    - '[+|-][X|Y][+|-][X|Y]'

    and it is then interpreted as above by appending '+Z'.

    Note that the string `sorting` is case insensitive.

    Example of written file::

        # [...]
        # # GRID - NUMBER OF CELLS
        # NX <int>
        # NY <int>
        # NZ <int>
        # # GRID - CELL SIZE
        # SX <float>
        # SY <float>
        # SZ <float>
        # # GRID - ORIGIN (bottom-lower-left corner)
        # OX <float>
        # OY <float>
        # OZ <float>
        # # GRID - FILLING
        # SORTING +X+Y+Z
        # [...]
        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (str) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    filename : str
        name of the file

    im : :class:`Img`
        image to be written

    sorting : str, default: '+X+Y+Z'
        describes the way to fill the grid (see above)

    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line

    comments : str, default:'#'
        string is used in the beginning of each line in the header, where grid
        geometry and sorting mode is written

    endofline : str, default: '\\\\n'
        end of line character

    usevars: 1D array-like or int, optional
        variable index(es) to be written; by default, all variables are written

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    fname = 'writeImageTxt'

    # Check comments identifier
    if comments is None or comments == '':
        err_msg = f'{fname}: `comments` cannot be an empty string (nor `None`)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if usevars is not None:
        if isinstance(usevars, int):
            if usevars < 0 or usevars >= im.nv:
                err_msg = f'{fname}: `usevars` invalid'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        else:
            if np.any([iv < 0 or iv >= im.nv for iv in usevars]):
                err_msg = f'{fname}: `usevars` invalid'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

    # Deal with sorting
    if len(sorting) == 4:
        sorting = sorting + '+Z'

    if len(sorting) != 6:
        err_msg = f'{fname}: `sorting` (string) invalid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    sorting = sorting.lower() # tranform to lower case
    s = sorting[1] + sorting[3]+ sorting[5]
    if s == 'xyz':
        tr = (1, 2, 3)
    elif s == 'xzy': # "transpose_xyz_to_xzy (inv. of "transpose_xyz_to_xzy") to do
        tr = (2, 1, 3)
    elif s == 'yxz': # "transpose_xyz_to_yxz (inv. of "transpose_xyz_to_yxz") to do
        tr = (1, 3, 2)
    elif s == 'yzx': # "transpose_xyz_to_zxy (inv. of "transpose_xyz_to_yzx") to do
        tr = (3, 1, 2)
    elif s == 'zxy': # "transpose_xyz_to_yzx (inv. of "transpose_xyz_to_zxy") to do
        tr = (2, 3, 1)
    elif s == 'zyx': # "transpose_xyz_to_zyx (inv. of "transpose_xyz_to_zyx") to do
        tr = (3, 2, 1)
    else:
        err_msg = f'{fname}: `sorting` (string) invalid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    flip = [1, 1, 1]
    for i in range(3):
        j = 2*i
        s1 = sorting[j]
        if s1 == '-':
            s2 = sorting[j+1]
            if s2 == 'x':
                flip[0] = -1
            elif s2 == 'y':
                flip[1] = -1
            else: # s2 == 'z'
                flip[2] = -1
        elif s1 != '+':
            err_msg = f'{fname}: `sorting` (string) invalid'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

    # Reorganize val array according to sorting, final shape: (im.nx*im.ny*im.nz, len(varname))
    val = im.val[:, ::flip[2], ::flip[1], ::flip[0]].transpose(0, *tr).reshape(-1, im.nx*im.ny*im.nz).T

    # Set header
    headerlines = []
    headerlines.append(f'{comments}')
    headerlines.append(f'{comments} # GRID - NUMBER OF CELLS')
    headerlines.append(f'{comments} NX {im.nx}')
    headerlines.append(f'{comments} NY {im.ny}')
    headerlines.append(f'{comments} NZ {im.nz}')
    headerlines.append(f'{comments} # GRID - CELL SIZE')
    headerlines.append(f'{comments} SX {im.sx}')
    headerlines.append(f'{comments} SY {im.sy}')
    headerlines.append(f'{comments} SZ {im.sz}')
    headerlines.append(f'{comments} # GRID - ORIGIN (bottom-lower-left corner)')
    headerlines.append(f'{comments} OX {im.ox}')
    headerlines.append(f'{comments} OY {im.oy}')
    headerlines.append(f'{comments} OZ {im.oz}')
    headerlines.append(f'{comments} # GRID - FILLING')
    headerlines.append(f'{comments} SORTING {sorting}')
    headerlines.append(f'{comments}')
    header = f'{endofline}'.join(headerlines) + f'{endofline}'

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        # Write header
        ff.write(header.encode())
        # Write variable values
        writeVarsTxt(ff, im.varname, val, missing_value=missing_value, delimiter=delimiter, usecols=usevars, fmt=fmt, logger=logger)

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readPointSetTxt(
        filename,
        missing_value=None,
        delimiter=' ',
        comments='#',
        usecols=None,
        set_xyz_as_first_vars=True,
        x_def=0.0, 
        y_def=0.0, 
        z_def=0.0,
        logger=None):
    """
    Reads a point set from a txt file.

    If the flag `set_xyz_as_first_vars=True`, the x, y, z coordinates of the
    points are set as variables with index 0, 1, 2, in the output point set
    (by reordering the variables if needed). The coordinates are identified by
    the names 'x', 'y', 'z' (case insensitive); if a coordinate is not present in
    the file, it is added as a variable in the output point set and set to the
    default value specified by `x_def`, `y_def`, `z_def` (for x, y, z) for all
    points.

    Example of file::

        # commented line ...
        # [...]
        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (string) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    filename : str
        name of the file

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line;
        note: "empty field" after splitting is ignored, then if white space is
        used as delimiter (default), one or multiple white spaces can be used as
        the same delimiter

    comments : str, default:'#'
        lines starting with that string are treated as comments

    usecols : 1D array-like or int, optional
        column index(es) to be read (first column corresponds to index 0);
        by default, all columns are read

    set_xyz_as_first_vars : bool
        - if `True`: the x, y, z coordinates are set as variables of index 0, 1, 2 \
        in the ouput point set (adding them and reodering if needed, see above)
        - if `False`: the variables of the point set will be the variables of the \
        columns read

    x_def : float, default: 0.0
        default values for x coordinates (used if x coordinate is added
        as variable and not read from the file and if `set_xyz_as_first_vars=True`)

    y_def : float, default: 0.0
        default values for y coordinates (used if y coordinate is added
        as variable and not read from the file and if `set_xyz_as_first_vars=True`)

    z_def : float, default: 0.0
        default values for z coordinates (used if z coordinate is added
        as variable and not read from the file and if `set_xyz_as_first_vars=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps : :class:`PointSet`
        point set (read from the file)
    """
    fname = 'readPointSetTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Read variale names and values from file
    try:
        varname, val = readVarsTxt(filename, missing_value=missing_value, delimiter=delimiter, comments=comments, usecols=usecols, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: variables names / values cannot be read'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg) from exc

    # Number of points and number of variables
    npt, nv = val.shape

    if set_xyz_as_first_vars:
        # Retrieve index of x, y, z coordinates
        ic = []

        ix = np.where([vn in ('x', 'X') for vn in varname])[0]
        if len(ix) == 1:
            ix = ix[0]
            ic.append(ix)
        elif len(ix) > 1:
            err_msg = f'{fname}: x-coordinates given more than once'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        else:
            ix = -1 # x-coordinates not given

        iy = np.where([vn in ('y', 'Y') for vn in varname])[0]
        if len(iy) == 1:
            iy = iy[0]
            ic.append(iy)
        elif len(iy) > 1:
            err_msg = f'{fname}: y-coordinates given more than once'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        else:
            iy = -1 # y-coordinates not given

        iz = np.where([vn in ('z', 'Z') for vn in varname])[0]
        if len(iz) == 1:
            iz = iz[0]
            ic.append(iz)
        elif len(iz) > 1:
            err_msg = f'{fname}: z-coordinates given more than once'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)

        else:
            iz = -1 # z-coordinates not given

        # Reorder varname and columns of val, if needed
        if not np.all([ic[i] == i for i in range(len(ic))]):
            # Reordering required
            ic = np.asarray(ic)
            ind = np.hstack((ic, np.setdiff1d(np.arange(nv), ic)))
            varname = list(np.asarray(varname)[ind])
            val = val[:, ind]

        # Insert missing coordinates, if needed
        if len(ic) < 3:
            if ix == -1:
                # Insert x-coordinates
                varname.insert(0, 'X')
                val = np.hstack((np.zeros(npt).reshape(-1, 1), val))
            if iy == -1:
                # Insert y-coordinates
                varname.insert(1, 'Y')
                val = np.hstack((val[:,:1], np.zeros(npt).reshape(-1, 1), val[:,1:]))
            if iz == -1:
                # Insert z-coordinates
                varname.insert(2, 'Z')
                val = np.hstack((val[:,:2], np.zeros(npt).reshape(-1, 1), val[:,2:]))

        # Update nv
        nv = len(varname)

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(val, val == missing_value, np.nan)

    # Set point set
    ps = PointSet(npt=npt, nv=nv, val=val.T, varname=varname)

    return ps
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writePointSetTxt(
        filename,
        ps,
        missing_value=None,
        delimiter=' ',
        comments='#',
        endofline='\n',
        usevars=None,
        fmt="%.10g",
        logger=None):
    """
    Writes a point set in a txt file.

    Example of file::

        #
        # # POINT SET - NUMBER OF POINTS AND NUMBER OF VARIABLES
        # NPT <int>
        # NV <int>
        #
        varname[0] varname[1] ... varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where `varname[j]` (string) is a the name of the variable of index j, and
    `v[i, j]` (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Parameters
    ----------
    filename : str
        name of the file

    ps : :class:`PointSet`
        point set to be written

    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    delimiter : str, default: ' '
        delimiter used to separate names and values in each line

    comments : str, default:'#'
        string is used in the beginning of each line in the header, where point
        set information is written

    endofline : str, default: '\\\\n'
        end of line character

    usevars: 1D array-like or int, optional
        variable index(es) to be written; by default, all variables are written

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    fname = 'writePointSetTxt'

    # Check comments identifier
    if comments is None or comments == '':
        err_msg = f'{fname}: `comments` cannot be an empty string (nor `None`)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if usevars is not None:
        if isinstance(usevars, int):
            if usevars < 0 or usevars >= ps.nv:
                err_msg = f'{fname}: `usevars` invalid'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

        else:
            if np.any([iv < 0 or iv >= ps.nv for iv in usevars]):
                err_msg = f'{fname}: `usevars` invalid'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

    # Set header
    headerlines = []
    headerlines.append(f'{comments}')
    headerlines.append(f'{comments} POINT SET')
    headerlines.append(f'{comments} NPT {ps.npt}')
    headerlines.append(f'{comments}')
    header = f'{endofline}'.join(headerlines) + f'{endofline}'

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        # Write header
        ff.write(header.encode())
        # Write variable values
        writeVarsTxt(ff, ps.varname, ps.val.T, missing_value=missing_value, delimiter=delimiter, usecols=usevars, fmt=fmt, logger=logger)

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImage2Drgb(
        filename,
        categ=False,
        nancol=None,
        keep_channels=True,
        rgb_weight=(0.299, 0.587, 0.114),
        flip_vertical=True,
        logger=None):
    """
    Reads an "RGB" image from a file.

    This function uses `matplotlib.pyplot.imread` to read the file, and fill a
    corresponding instance of :class:`Img`. The file represents a 2D image, with
    a RGB or RGBA code for every pixel, the file format can be png, ppm, jpeg,
    etc. (e.g. created by Gimp).
    Note that every channel (RGB) is renormalized in [0, 1] by dividing by 255
    if needed.

    Treatement of colors (RGB or RGBA):

    - `nancol` is a color (RGB or RGBA) that is considered as "missing value", \
    i.e. `numpy.nan` in the output image,
    - if `keep_channels=True`: every channel is retrieved (3 channels for RGB \
    or 4 channels for RGBA)
    - if `keep_channels=False`: the channels RGB (alpha channel, if present, \
    is ignored) are linearly combined using the weights `rgb_weight`, to get \
    color codes defined as one value in [0, 1]

    Type of image:

    - continuous (`categ=False`): the output image has one variable if \
    `keep_channels=False`, and 3 or 4 variables (resp. for colors as RGB or \
    RGBA codes) if `keep_channels=True`
    - categorical (`categ=True`): the list of distinct colors is retrieved (`col`) \
    and indexed (from 0); the ouptut image has one variable defined as the \
    index of the color (in the list `col`); the list `col` is also retrieved \
    in output, every entry is a unique value (`keep_channels=False`) or a \
    sequence of length 3 or 4 (`keep_channels=True`). Note that the output \
    image can be displayed (plotted) directly by using:
        - `geone.imgplot.drawImage2D(im, categ=True, categCol=col)`, \
        if `keep_channels=True`
        - `geone.imgplot.drawImage2D(im, categ=True, categCol=[cmap(c) for c in col])`, \
        where cmap is a color map function defined on the interval [0, 1], \
        if `keep_channels=False`

    Parameters
    ----------
    filename : str
        name of the file

    categ : bool, default: False
        indicattes the type of output image

        - if `True`: "categorical" output image with one variable interpreted as \
        an index (see above)
        - if `False`: "continuous" output image

    nancol : color, optional
        color (RGB color code (alpha channel, if present, is ignored) or str)
        interpreted as missing value (`numpy.nan`) in output image

    keep_channels : bool, default: True
        for RGB or RGBA images

        - if `True`: every channel are retrieved
        - if `False`: first three channels (RGB) are linearly combined using the \
        weight `rgb_weight`, to define one variable (alpha channel, if present, \
        is ignored)

    rgb_weight : 1D array-like of 3 floats
        weights for R, G, B channels used to combine channels (if
        `keep_channels=False`);
        notes:

        - by default: values set from `Pillow`, image convert mode “L”
        - other weights can be e.g. (0.2125, 0.7154, 0.0721)

    flip_vertical : bool, default: True
        indicates if the image is flipped vartically after reading (this is useful
        because the "origin" of the input image is considered at the top left
        (using `matplotlib.pyplot.imread`), whereas it is at bottom left in the
        output image)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im : :class:`Img`
        output image (see "Type of image" above)

    col : list, optional
        list of colors, each component is a unique value (in [0,1]) or a 3-tuple
        (RGB code) or a 4-tuple (RGBA code); the output image has one variable
        which is the index of the color;
        returned if `categ=True`
    """
    fname = 'readImage2Drgb'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Read image
    vv = plt.imread(filename)

    # Reshape image: one pixel per line
    ny, nx = vv.shape[0:2]
    vv = vv.reshape(nx*ny, -1)
    nv = vv.shape[1]

    # Check input image
    if nv != 3 and nv != 4:
        err_msg = f'{fname}: the input image must be in RGB or RGBA (3 or 4 channels for each pixel)'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Normalize channels if needed
    if vv.dtype == 'uint8':
        vv = vv/255.

    if nancol is not None:
        # "format" nancol (ignoring alpha channel)
        nancolf = mcolors.to_rgb(nancol)
        ind_missing = np.where([np.all(vvi[:3]==nancolf) for vvi in vv])
        vv = vv.astype('float') # to ensure ability to set np.nan in vv
        vv[ind_missing, :] = np.nan

    if categ:
        ind_isnan = np.any(np.isnan(vv), axis=1)
        v = np.repeat(np.nan, vv.shape[0])
        col, v[~ind_isnan] = np.unique(vv[~ind_isnan], axis=0, return_inverse=True)
        vv = v
        if keep_channels:
            col = list(col)
        else:
            col = col[:,0:3].dot(rgb_weight)
        nv = 1
        varname = 'code'

    else:
        if keep_channels:
            if nv == 3:
                varname = ['red', 'green', 'blue']
            elif nv == 4:
                varname = ['red', 'green', 'blue', 'alpha']
            vv = vv.T
        else:
            # Set value for each pixel
            #ind_isnan = np.any(np.isnan(vv), axis=1)
            #v = np.repeat(np.nan, vv.shape[0])
            #v[~ind_isnan] = vv[~ind_isnan,0:3].dot(rgb_weight) # ignore alpha channel if present
            #vv = v
            vv = vv[:,0:3].dot(rgb_weight) # ignore alpha channel if present
            nv = 1
            varname = 'val'

    if flip_vertical:
        vv = vv.reshape(nv, ny, nx)
        vv = vv[:,::-1,:] # vertical flip

    # Set output image
    #im = Img(nx, ny, 1, nv=nv, val=vv, varname=varname)
    im = Img(nx, ny, 1, nv=nv, val=vv, varname=varname, logger=logger)

    if categ:
        out = (im, col)
    else:
        out = im

    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImage2Drgb(
        filename,
        im,
        col=None,
        cmap='gray',
        nancol=(1.0, 0.0, 0.0),
        flip_vertical=True,
        verbose=0,
        logger=None):
    """
    Writes (saves) an "RGB" image in a file.

    This function uses `matplotlib.pyplot.imsave`, to write a file in format:
    png, ppm, jpeg, etc. The input image `im` (:class:`Img`) must be in 2D (i.e.
    `im.nz=1`) with one variable, 3 variables (channels RGB) or 4 variables
    (channels RGBA).

    Treatement of colors (RGB or RGBA):

    - if the input image has one variable, then:
        - if a list `col` of colors (RGB, RGBA, or str) is given: the image \
        variable must be an index (integer) in  {0, ..., len(col)-1}, then the \
        colors from the list are used according to the index (variable value) at \
        each pixel
        - if `col=None` (not given, default), the colors are set from the \
        variable by using colormap `cmap`, the variable values should be floats \
        in the interval [0,1]
    - if the input image has 3 or 4 variables, they are interpreted as RGB \
    or RGBA color codes
    - `nancol` is the color used for missing value (`numpy.nan`) in input image

    Parameters
    ----------
    filename : str
        name of the file

    im : :class:`Img`
        input image

    col : 1D array-like of object representing color, optional
        sequence of colors (3-tuple for RGB code, 4-tuple for RGBA code, str)
        used for each category (index) of the input image, used only if the input
        image has one variable (with value in {0, 1, ..., len(col)-1})

    cmap : colormap
        color map (can be a string, in this case the color map
        `matplotlib.pyplot.get_cmap(cmap)` is used, only for image with one
        variable when `col=None`

    nancol : color, default: (1.0, 0.0, 0.0)
        color (3-tuple for RGB code, 4-tuple for RGBA code, str) used for missing
        value (`numpy.nan`) in the input image

    flip_vertical : bool, default: True
        indicates if the image is flipped vartically before writing (this is
        useful because the "origin" of the input image is considered at the
        bottom left, whereas it is at top left in file png, etc.)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    fname = 'writeImage2Drgb'

    # Check image parameters
    if im.nz != 1:
        err_msg = f'{fname}: `im.nz` must be 1'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    if im.nv not in [1, 3, 4]:
        err_msg = f'{fname}: `im.nv` must be 1, 3, or 4'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Extract the array of values
    vv = np.copy(im.val) # copy to not modify original values

    if flip_vertical:
        vv = vv[:,:,::-1,:]

    # Reshape and transpose the array of values
    vv = vv.reshape(im.nv, -1).T

    if vv.shape[1] == 1: # im.nv == 1
        if col is not None:
            try:
                nchan = len(col[0])
            except:
                err_msg = f'{fname}: `col` must be a sequence of RGB or RBGA color (each entry is a sequence of length 3 or 4)'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            if not np.all(np.array([len(c) for c in col]) == nchan):
                err_msg = f'{fname}: same format is required for every color in `col`'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            # "format" nancol
            if nchan == 3:
                nancolf = mcolors.to_rgb(nancol)
            elif nchan == 4:
                nancolf = mcolors.to_rgba(nancol)
            else:
                err_msg = f'{fname}: invalid format for the colors (RGB or RGBA required)'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            # Check value in vv
            if np.any((vv < 0, vv >= len(col))):
                err_msg = f'{fname}: variable value in image cannot be treated as index in `col`'
                if logger: logger.error(err_msg)
                raise ImgError(err_msg)

            # Set ouput colors
            vv = np.array([col[int(v)] if ~np.isnan(v) else nancolf for v in vv.reshape(-1)])

        else:
            # "format" nancol
            nancolf = mcolors.to_rgba(nancol)

            # Set ouput colors (grayscale, coded as rgb)
            # Get the color map
            if isinstance(cmap, str):
                try:
                    cmap = plt.get_cmap(cmap)
                except:
                    err_msg = f'{fname}: invalid `cmap` string'
                    if logger: logger.error(err_msg)
                    raise ImgError(err_msg)

            if np.any((vv < 0, vv > 1)):
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: variable values in image are not in interval [0,1], they are rescaled')
                    else:
                        print(f'{fname}: WARNING: variable values in image are not in interval [0,1], they are rescaled')

                ind = np.where(~np.isnan(vv))
                vmin = np.min(vv[ind])
                vmax = np.max(vv[ind])
                vv = (vv - vmin)/(vmax - vmin)

            vv = np.array([cmap(v) if ~np.isnan(v) else nancolf for v in vv.reshape(-1)])

    else: # vv.shape[1] is 3 or 4
        # "format" nancol
        if vv.shape[1] == 3:
            nancolf = mcolors.to_rgb(nancol)
        else: # vv.shape[1] == 4
            nancolf = mcolors.to_rgba(nancol)

        ind_isnan = np.any(np.isnan(vv), axis=1)
        vv[ind_isnan, :] = nancolf

    # Format the array of values
    vv = np.ascontiguousarray(vv.reshape(im.ny, im.nx, -1))

    # Save the image in file
    plt.imsave(filename, vv)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageGslib(filename, missing_value=None, logger=None):
    """
    Reads an image from a file in "gslib" format.

    It is recommended to use the functions
    :func:`readImageTxt` / :func:`writeImageTxt` instead.

    File is assumed to be in the following format (text file)::

        Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
        nv
        varname[0]
        ...
        varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[Nx*Ny*Nz-1, 0]  v[n-1, 1]  ... v[Nx*Ny*Nz-1-1, nv-1]
        V1(0)    ... Vnvar(0)
        ...
        V1(n-1) ... Vnvar(n-1)

    where

    - Nx, Ny, Nz are the number of grid cell along x, y, z axes
    - Sx, Sy, Sz are the cell size along x, y, z axes (default 1.0)
    - Ox, Oy, Oz are the coordinates of the (bottom-lower-left corner) \
    (default (0.0, 0.0, 0.0))
    - `varname[j]` (string) is a the name of the variable of index j, and \
    `v[i, j]` (float) is the value of the variable of index j, for the entry of \
    index i, i.e. one entry per line; the grid is filled (with values as they \
    are written) with \
    x index increases, then y index increases, then z index increases

    Parameters
    ----------
    filename : str
        name of the file

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im : :class:`Img`
        image (read from the file)
    """
    fname = 'readImageGslib'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read 1st line
        line1 = ff.readline()

        # Read 2nd line
        line2 = ff.readline()

        # Set number of variables
        nv = int(line2)

        # Set variable name (next nv lines)
        varname = [ff.readline().replace("\n",'') for i in range(nv)]

        # Read the rest of the file
        valarr = np.loadtxt(ff)

    # Convert line1 as list
    g = [x for x in line1.split()]

    # Set grid
    nx, ny, nz = [int(n) for n in g[0:3]]
    sx, sy, sz = [1.0, 1.0, 1.0]
    ox, oy, oz = [0.0, 0.0, 0.0]

    if len(g) >= 6:
        sx, sy, sz = [float(n) for n in g[3:6]]

    if len(g) >= 9:
        ox, oy, oz = [float(n) for n in g[6:9]]

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr.T, varname, filename, logger=logger)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageGslib(im, filename, missing_value=None, fmt="%.10g"):
    """
    Writes an image in a file in "gslib" format.

    It is recommended to use the functions
    :func:`readImageTxt` / :func:`writeImageTxt` instead.

    File is written in the following format (text file)::

        Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
        nv
        varname[0]
        ...
        varname[nv-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[Nx*Ny*Nz-1, 0]  v[n-1, 1]  ... v[Nx*Ny*Nz-1-1, nv-1]
        V1(0)    ... Vnvar(0)
        ...
        V1(n-1) ... Vnvar(n-1)

    where
    - Nx, Ny, Nz are the number of grid cell along x, y, z axes
    - Sx, Sy, Sz are the cell size along x, y, z axes (default 1.0)
    - Ox, Oy, Oz are the coordinates of the (bottom-lower-left corner) \
    (default (0.0, 0.0, 0.0))
    - `varname[j]` (string) is a the name of the variable of index j, and \
    `v[i, j]` (float) is the value of the variable of index j, for the entry of \
    index i, i.e. one entry per line; the grid is filled (with values as they \
    are written) with \
    x index increases, then y index increases, then z index increases

    Parameters
    ----------
    im : :class:`Img`
        image to be written

    filename : str
        name of the file

    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    # fname = 'writeImageGslib'

    # Write 1st line in string shead
    shead = "{} {} {}   {} {} {}   {} {} {}\n".format(
            im.nx, im.ny, im.nz, im.sx, im.sy, im.sz, im.ox, im.oy, im.oz)
    # Append 2nd line
    shead = shead + "{}\n".format(im.nv)

    # Append variable name(s) (next line(s))
    for s in im.varname:
        shead = shead + "{}\n".format(s)

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(im.val, np.isnan(im.val), missing_value)

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        ff.write(shead.encode())
        # Write variable values
        np.savetxt(ff, im.val.reshape(im.nv, -1).T, delimiter=' ', fmt=fmt)

    # Replace missing_value by np.nan (restore)
    if missing_value is not None:
        np.putmask(im.val, im.val == missing_value, np.nan)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageVtk(filename, missing_value=None, logger=None):
    """
    Reads an image from a file in "vtk" format.

    It is recommended to use the functions
    :func:`readImageTxt` / :func:`writeImageTxt` instead.

    Parameters
    ----------
    filename : str
        name of the file

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im : :class:`Img`
        image (read from the file)
    """
    fname = 'readImageVtk'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read lines 1 to 10
        header = [ff.readline() for i in range(10)]

        # Read the rest of the file
        valarr = np.loadtxt(ff)

    # Set grid
    nx, ny, nz = [int(n) for n in header[4].split()[1:4]]
    sx, sy, sz = [float(n) for n in header[6].split()[1:4]]
    ox, oy, oz = [float(n) for n in header[5].split()[1:4]]

    # Set variable
    tmp = header[8].split()
    if len(tmp) > 3:
        nv = int(tmp[3])
    else:
        nv = 1
    varname = tmp[1].split('/')

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr.T, varname, filename, logger=logger)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageVtk(
        im,
        filename,
        missing_value=None,
        fmt="%.10g",
        data_type='float',
        version=3.4,
        name=None):
    """
    Writes an image in a file in "vtk" format.

    It is recommended to use the functions
    :func:`readImageTxt` / :func:`writeImageTxt` instead.

    Parameters
    ----------
    im : :class:`Img`
        image to be written

    filename : str
        name of the file

    missing_value : float, optional
        if specified: `numpy.nan` value will be replaced by `missing_value`;
        otherwise, `numpy.nan` will be used

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    data_type : str, default: 'float'
        data type (of the image variables)

    version : str, default: '3.4'
        version number (written in vtk data file)

    name : str, optional
        name to be written at line 2; by default (`None`): `im.name` is used

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    # fname = 'writeImageVtk'

    if name is None:
        name = im.name

    # Set header (10 first lines)
    shead = (
            "# vtk DataFile Version {0}\n"
            "{1}\n"
            "ASCII\n"
            "DATASET STRUCTURED_POINTS\n"
            "DIMENSIONS {2} {3} {4}\n"
            "ORIGIN     {5} {6} {7}\n"
            "SPACING    {8} {9} {10}\n"
            "POINT_DATA {11}\n"
            "SCALARS {12} {13} {14}\n"
            "LOOKUP_TABLE default\n"
        ).format(version,
                 name,
                 im.nx, im.ny, im.nz,
                 im.ox, im.oy, im.oz,
                 im.sx, im.sy, im.sz,
                 im.nxyz(),
                 '/'.join(im.varname), data_type, im.nv)

    # Save data into a separate array, otherwise the img will be modified
    data = np.copy(im.val)

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(data, np.isnan(data), missing_value)

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        ff.write(shead.encode())
        # Write variable values
        np.savetxt(ff, data.reshape(im.nv, -1).T, delimiter=' ', fmt=fmt)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageGrd(filename, varname='V0', logger=None):
    """
    Reads an image (2D, one variable) from a file in "grd" (or "asc") format.

    The written file has the header:

        ncols <nx>
        nrows <ny>
        xllcorner <ox>
        yllcorner <oy>
        cellsize <resolution>
        NODATA_value <missing_value>

    with the values of one variable starting from the upper left grid cell, 
    from left to right (along columns or x-axis), then from top to bottom (along 
    rows or y-axis), one value per line, with <missing_value> for no data entries
    (that will be replaced by `numpy.nan` in the output image).

    Parameters
    ----------
    filename : str
        name of the file, should has the extension '.grd' or '.asc'

    varname : str, default: 'V0'
        name of the variable in the output image

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im : :class:`Img`
        image (read from the file)
    """
    fname = 'readImageGrd'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read lines 1 to 6
        header = [ff.readline() for i in range(6)]

        # Read the rest of the file
        valarr = np.loadtxt(ff)

    # Get grid and missing value
    nx, ny, ox, oy, sx, missing_value = None, None, None, None, None, None
    for i in range(6):
        k, v = header[i].split()
        if k == 'ncols':
            nx = int(v)
        elif k == 'nrows':
            ny = int(v)
        elif k == 'xllcorner':
            ox = float(v)
        elif k == 'yllcorner':
            oy = float(v)
        elif k == 'cellsize':
            sx = float(v)
        elif k == 'NODATA_value':
            missing_value = float(v)
        else:
            err_msg = f'{fname}: invalid "key" in header of the file'
            if logger: logger.error(err_msg)
            raise ImgError(err_msg)
    
    if nx is None or ny is None or ox is None or oy is None or sx is None or missing_value is None:
        err_msg = f'{fname}: invalid file header'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Set value of the variable
    if valarr.size != nx*ny:
        err_msg = f'{fname}: invalid number of values in the file'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Replace missing_value by np.nan
    np.putmask(valarr, valarr == missing_value, np.nan)
    
    # Reshape and flip along y axis
    valarr = valarr.reshape(ny, nx)[::-1, :]

    # Set image
    sy = sx
    nz = 1
    sz = 1.0
    oz = 0.0

    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, 1, valarr, varname, filename, logger=logger)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageGrd(
        filename,
        im,
        iv=0,
        iz=0,
        missing_value=99999,
        endofline='\n',
        fmt="%.10g",
        logger=None):
    """
    Writes an image layer in a text file (format / extention grd or asc, e.g. for QGis).

    One variable in one layer (constant z index) is written.

    The cell size along x and y axes must be the same (resolution).

    The written file has the header:

        ncols <nx>
        nrows <ny>
        xllcorner <ox>
        yllcorner <oy>
        cellsize <resolution>
        NODATA_value <missing_value>

    with the values of one variable on a given z-layer, starting from the upper
    left grid cell, from left to right (along columns or x-axis), then from top to 
    bottom (along rows or y-axis), one value per line, with <missing_value> for
    np.nan entries.

    Parameters
    ----------
    filename : str
        name of the file, should has the extension '.grd' or '.asc'

    im : :class:`Img`
        image to be written

    iv: int, default: 0
        index of the variable to be written

    iz: int, default: 0
        index of the z-layer in the image (along z-axis) to be written
    
    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    endofline : str, default: '\\\\n'
        end of line character

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    fname = 'writeImageGrd'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        err_msg = f'{fname}: variable index `iv` not valid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Check iz
    if iz < 0:
        iz = im.nz + iz

    if iz < 0 or iz >= im.nz:
        err_msg = f'{fname}: layer index `iz` not valid'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Check resolution (im.sx and im.sy must be equal)
    if not np.isclose(im.sx, im.sy):
        err_msg = f'{fname}: resolution of image not valid: cell size in x and y direction (`sx`, `sy`) must be equal'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Extract values (flip along y axis)
    val = im.val[iv, iz, ::-1, :].reshape(-1)

    # Change np.nan to missing_value
    np.putmask(val, np.isnan(val), missing_value)

    # Set header
    headerlines = []
    headerlines.append(f'ncols {im.nx}')
    headerlines.append(f'nrows {im.ny}')
    headerlines.append(f'xllcorner {im.ox}')
    headerlines.append(f'yllcorner {im.oy}')
    headerlines.append(f'cellsize {im.sx}')
    headerlines.append(f'NODATA_value {missing_value}')
    header = f'{endofline}'.join(headerlines)

    np.savetxt(filename, val, comments='', header=header, fmt=fmt)

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readPointSetGslib(filename, missing_value=None, logger=None):
    """
    Reads a point set from a file in "gslib" format.

    It is recommended to use the functions
    :func:`readPointSetTxt` / :func:`writePointSetTxt` instead.

    File is assumed to be in the following format::

        NPT
        NV
        varname[0]
        ...
        varname[NV-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where
    - NPT is the number of points
    - NV is the number of variable, including points coordinates (location)
    - `varname[j]` (string) is a the name of the variable of index j, and `v[i, j]` \
    (float) is the value of the variable of index j, for the entry of index i, \
    i.e. one entry per line.

    Parameters
    ----------
    filename : str
        name of the file

    missing_value : float, optional
        value that will be replaced by `numpy.nan`

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps : :class:`PointSet`
        point set (read from the file)
    """
    fname = 'readPointSetGslib'

    # Check if the file exists
    if not os.path.isfile(filename):
        err_msg = f'{fname}: invalid filename ({filename})'
        if logger: logger.error(err_msg)
        raise ImgError(err_msg)

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read 1st line
        line1 = ff.readline()

        # Read 2nd line
        line2 = ff.readline()

        # Set number of variables
        nv = int(line2)

        # Set variable name (next nv lines)
        varname = [ff.readline().replace("\n",'') for i in range(nv)]

        # Read the rest of the file
        valarr = np.loadtxt(ff)

    # Set number of point(s)
    npt = int(line1)

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set point set
    ps = PointSet(npt=npt, nv=nv, val=valarr.T, varname=varname)

    return ps
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writePointSetGslib(ps, filename, missing_value=None, fmt="%.10g"):
    """
    Writes a point set in a file in "gslib" format.

    It is recommended to use the functions
    :func:`readPointSetTxt` / :func:`writePointSetTxt` instead.

    File is written in the following format::

        NPT
        NV
        varname[0]
        ...
        varname[NV-1]
        v[0, 0]    v[0, 1]    ... v[0, nv-1]
        v[1, 0]    v[1, 1]    ... v[1, nv-1]
        ...
        v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]

    where
    - NPT is the number of points
    - NV is the number of variable, including points coordinates (location)
    - `varname[j]` (string) is a the name of the variable of index j, and `v[i, j]` \
    (float) is the value of the variable of index j, for the entry of index i, \
    i.e. one entry per line.

    Parameters
    ----------
    ps : :class:`PointSet`
        point set to be written

    filename : str
        name of the file

    missing_value : float, optional
        `numpy.nan` value will be replaced by `missing_value` before writing

    fmt : str, default: '%.10g'
        format for single variable value, `fmt` is a string of the form
        '%[flag]width[.precision]specifier'

    Notes
    -----
    For more details about format (`fmt` parameter), see
    https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    # fname = 'writePointSetGslib'

    # Write 1st line in string shead
    shead = "{}\n".format(ps.npt)

    # Append 2nd line
    shead = shead + "{}\n".format(ps.nv)

    # Append variable name(s) (next line(s))
    for s in ps.varname:
        shead = shead + "{}\n".format(s)

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(ps.val, np.isnan(ps.val), missing_value)

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        ff.write(shead.encode())
        # Write variable values
        np.savetxt(ff, ps.val.reshape(ps.nv, -1).T, delimiter=' ', fmt=fmt)

    # Replace missing_value by np.nan (restore)
    if missing_value is not None:
        np.putmask(ps.val, ps.val == missing_value, np.nan)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.img' example:")
    print("   See jupyter notebook for examples...")
