#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'img.py'
author:         Julien Straubhaar
date:           jan-2018

Definition of classes for images and point sets, and relative functions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================================
class Img(object):
    """
    Defines an image as a 3D grid with variable(s) / attribute(s):
        nx, ny, nz: (int) number of grid cells along each axis
        sx, sy, sz: (float) cell size along each axis
        ox, oy, oz: (float) origin of the grid (bottom-lower-left corner)
        nv:         (int) number of variable(s) / attribute(s)
        val:        ((nv,nz,ny,nx) array) attribute(s) / variable(s) values
        varname:    (list of string (or string)) variable names
        name:       (string) name of the image
    """

    def __init__(self,
                 nx=1,   ny=1,   nz=1,
                 sx=1.0, sy=1.0, sz=1.0,
                 ox=0.0, oy=0.0, oz=0.0,
                 nv=0, val=np.nan, varname=None,
                 name=""):
        """
        Init function for the class:

        :param val: (int/float or tuple/list/ndarray) value(s) of the new
                        variable:
                        if type is int/float: constant variable
                        if tuple/list/ndarray: must contain nv*nx*ny*nz values,
                            which are put in the image (after reshape if
                            needed)
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

        valarr = np.asarray(val, dtype=float)  # possibly 0-dimensional
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(nx*ny*nz*nv)
        elif valarr.size != nx*ny*nz*nv:
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        self.val = valarr.reshape(nv, nz, ny, nx)

        if varname is None:
            self.varname = ["V{:d}".format(i) for i in range(nv)]
        else:
            varname = list(np.asarray(varname).reshape(-1))
            if len(varname) == nv:
                self.varname = varname
            elif len(varname) == 1: # more than one variable and only one varname
                self.varname = ["{}{:d}".format(varname[0], i) for i in range(nv)]
            else:
                print(f'ERROR ({fname}): varname has not an acceptable size')
                return None

        self.name = name

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
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
        Sets default variable names: varname = ('V0', 'V1',...).
        """
        self.varname = ["V{:d}".format(i) for i in range(self.nv)]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_varname(self, varname=None, ind=-1):
        """
        Sets name of the variable of the given index (if varname is None:
        'V' appended by the variable index is used as varame).
        """
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        if varname is None:
            varname = "V{:d}".format(ii)
        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_dimension(self, nx, ny, nz, newval=np.nan):
        """
        Sets dimensions and update shape of values array (by possible
        truncation or extension).

        :param nx, ny, nz:  (int) dimensions (number of cells) in x, y, z
                                direction
        :param newval:      (float) new value to insert if the array of values
                                has to be extended
        """

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
    def set_spacing(self, sx, sy, sz):
        """
        Sets cell size (sx, sy, sz).
        """
        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_origin(self, ox, oy, oz):
        """
        Sets grid origin (ox, oy, oz).
        """
        self.ox = float(ox)
        self.oy = float(oy)
        self.oz = float(oz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_grid(self, nx, ny, nz, sx, sy, sz, ox, oy, oz, newval=np.nan):
        """
        Sets grid (dimension, cell size, and origin).
        """
        self.set_dimension(nx, ny, nz, newval)
        self.set_spacing(sx, sy, sz)
        self.set_origin(ox, oy, oz)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def resize(self,
               ix0=0, ix1=None,
               iy0=0, iy1=None,
               iz0=0, iz1=None,
               iv0=0, iv1=None,
               newval=np.nan,
               newvarname=""):
        """
        Resizes the image.
        According to the x(, y, z) direction, the slice from ix0 to ix1-1
        (iy0 to iy1-1, iz0 to iz1-1) is considered (if None, ix1(, iy1, iz1)
        is set to nx(, ny, nz)), deplacing the origin from ox(, oy, oz)
        to ox+ix0*sx(, oy+iy0*sy, oz+iz0*sz), and inserting value newval at
        possible new locations:

        :param ix0, ix1:    (int or None) indices for x direction ix0 < ix1
        :param iy0, iy1:    (int or None) indices for y direction iy0 < iy1
        :param iz0, iz1:    (int or None) indices for z direction iz0 < iz1
        :param iv0, iv1:    (int or None) indices for v direction iv0 < iv1
        :param newval:      (float) new value to insert at possible new location
        :param newvarname:  (string) prefix for new variable name(s)
        """

        if ix1 is None:
            ix1 = self.nx

        if iy1 is None:
            iy1 = self.ny

        if iz1 is None:
            iz1 = self.nz

        if iv1 is None:
            iv1 = self.nv

        if ix0 >= ix1:
            print("Nothing is done! (invalid indices along x)")
            return None

        if iy0 >= iy1:
            print("Nothing is done! (invalid indices along y)")
            return None

        if iz0 >= iz1:
            print("Nothing is done! (invalid indices along z)")
            return None

        if iv0 >= iv1:
            print("Nothing is done! (invalid indices along v)")
            return None

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
        # n0 = -np.min([iv0, 0])           # number of new variable(s) to prepend
        # n1 = np.max([iv1-initshape[0], 0]) # number of new variable(s) to append
        self.varname = ['newvarname' + '{}'.format(i) for i in range(n0[0])] +\
                       [self.varname[i]
                        for i in range(np.max([iv0, 0]), np.min([iv1, initShape[0]]))] +\
                       ['newvarname' + '{}'.format(n0[0]+i) for i in range(n1[0])]

        # Update nx, ny, nz, nv
        self.nx = self.val.shape[3]
        self.ny = self.val.shape[2]
        self.nz = self.val.shape[1]
        self.nv = self.val.shape[0]

        # Update ox, oy, oz
        self.ox = self.ox + ix0 * self.sx
        self.oy = self.oy + iy0 * self.sy
        self.oz = self.oz + iz0 * self.sz
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, val=np.nan, varname=None, ind=0):
        """
        Inserts one or several variable(s) at a given index.

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: one constant variable is
                                inserted
                            if tuple/list/ndarray: its size must be a multiple
                                of self.nx*self.ny*self.nz
        :param varname: (string, list or 1-d array of strings or None)
                            name(s) of the new variable(s), if not given (None),
                            default variable names are set ("V<num>", where
                            <num> starts from the number of variables before
                            inserting)
        :param ind:     (int) index where the new variable(s) is (are) inserted
        """

        fname = 'insert_var'

        # Check / set ind
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            print("Nothing is done! (invalid index)")
            return None

        # Check val, set valarr (array of values)
        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size % self.nxyz() != 0:
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        m = valarr.size // self.nxyz() # number of variable(s) to be inserted

        # Check / set varname
        if varname is not None:
            if isinstance(varname, str):
                varname = [varname]
            elif (not isinstance(varname, tuple) and not isinstance(varname, list) and not (isinstance(varname, np.ndarray) and im_list.ndim==1)) or len(varname)!=m:
                print(f'ERROR ({fname}): varname does not have an acceptable size')
                return None
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
    def append_var(self, val=np.nan, varname=None):
        """
        Appends one or several variable(s).

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: one constant variable is
                                inserted
                            if tuple/list/ndarray: its size must be a multiple
                                of self.nx*self.ny*self.nz
        :param varname: (string, list or 1-d array of strings or None)
                            name(s) of the new variable(s), if not given (None),
                            default variable names are set ("V<num>", where
                            <num> starts from the number of variables before
                            appending)
        """

        self.insert_var(val=val, varname=varname, ind=self.nv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=None, indlist=None):
        """
        Removes variable(s) (of given index-es).

        :param ind:     (int or list of ints) index or list of index-es of the
                            variable(s) to be removed

        :param indlist: used for ind if ind is not given (None)
                            (obsolete, kept for compatibility with older
                                versions)
        """

        if ind is None:
            ind = indList
            if ind is None:
                print("Nothing is done! (no index given)")
                return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            print("Nothing is done! (invalid index)")
            return

        ind = np.setdiff1d(np.arange(self.nv), ind)

        self.extract_var(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """
        Removes all variables.
        """

        # Update val array
        del (self.val)
        self.val = np.zeros((0, self.nz, self.ny, self.nx))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, val=np.nan, varname=None, ind=-1):
        """
        Sets one variable (of given index).

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain nx*ny*nz values
        :param varname: (string or None) name of the variable
        :param ind:     (int) index where the variable to be set
        """

        fname = 'set_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size != self.nxyz():
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.nz, self.ny, self.nx)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, ind=None, indlist=None):
        """
        Extracts variable(s) (of given index-es).
        (May be used for reordering / duplicating variables.)

        :param ind:     (int or list of ints) index or list of index-es of the
                            variable(s) to be extracted (kept)

        :param indlist: used for ind if ind is not given (None)
                            (obsolete, kept for compatibility with older
                                versions)
        """

        if ind is None:
            ind = indList
            if ind is None:
                print("Nothing is done! (no index given)")
                return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allvar()
            return

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        self.val = self.val[ind,...]

        # Update varname list
        self.varname = [self.varname[i] for i in ind]

        # Update nv
        self.nv = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique_one_var(self, ind=0, ignore_missing_value=True):
        """
        Gets unique values of one variable (of given index).

        :param ind: (int) index of the variable
        :param ignore_missing_value:
                    (bool) if True: missing value (nan), if present, are ignored;
                        if False: missing value (nan), if present, is retrieved in
                        output

        :return:    (1-dimensional array) unique values of the variable
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        uval = np.unique(self.val[ii])

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True, ignore_missing_value=True):
        """
        Gets proportions (density or count) of unique values of one
        variable (of given index).

        :param ind:     (int) index of the variable
        :param density: (bool) computes densities if True and counts otherwise
        :param ignore_missing_value:
                        (bool) if True: missing value (nan), if present, are
                            ignored; if False: missing value (nan), if present,
                            is taken into account

        :return out:    (2-tuple of 1-dimensional array)
                            out[0]: (1-dimensional array) unique values of
                                    the variable
                            out[1]: (1-dimensional array) densities or counts of
                                    the unique values
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        uv, cv = np.unique(self.val[ii], return_counts=True)

        if ignore_missing_value:
            ind_known = ~np.isnan(uv)
            uv = uv[ind_known]
            cv = cv[ind_known]

        if density:
            cv = cv / np.sum(cv)

        return (uv, cv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique(self, ignore_missing_value=True):
        """
        Gets unique values among all variables.

        :param ignore_missing_value:
                    (bool) if True: missing value (nan), if present, are ignored;
                        if False: missing value (nan), if present, is retrieved in
                        output

        :return:    (1-dimensional array) unique values found in any variable
        """

        uval = np.unique(self.val)

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop(self, density=True, ignore_missing_value=True):
        """
        Gets proportions (density or count) of unique values for each variable.

        :param density: (bool) computes densities if True and counts otherwise
        :param ignore_missing_value:
                        (bool) if True: missing value (nan), if present, are
                            ignored; if False: missing value (nan), if present,
                            is taken into account

        :return out:    (list (of length 2) of 1-dimensional array)
                            out[0]: (1-dimensional array) unique values found in
                                any variable
                            out[1]: ((self.nv, len(out[0])) array) densities or
                                counts of the unique values:
                                out[i, j]: density or count of the j-th unique
                                    value for the i-th variable
        """

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
            uv, cv = self.get_prop_one_var(ind=i, density=density, ignore_missing_value=ignore_missing_value)
            for j, v in enumerate(uv):
                if np.isnan(v):
                    cv_all[i, uv_all_ind_nan] = cv[j]
                else:
                    cv_all[i, uv_all==v] = cv[j]

        return (uv_all, cv_all)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipx(self):
        """
        Flips variable values according to x direction.
        """
        self.val = self.val[:,:,:,::-1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipy(self):
        """
        Flips variable values according to y direction.
        """
        self.val = self.val[:,:,::-1,:]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipz(self):
        """
        Flips variable values according to z direction.
        """
        self.val = self.val[:,::-1,:,:]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def flipv(self):
        """
        Flips variable values according to v direction.
        """
        self.val = self.val[::-1,:,:,:]
        self.varname = self.varname[::-1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permxy(self):
        """
        Permutes / swaps x and y directions.
        (Obsolete, use swap_xy)
        """
        self.swap_xy()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permxz(self):
        """
        Permutes / swaps x and z directions.
        (Obsolete, use swap_xz)
        """
        self.swap_xz()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permzy(self):
        """
        Permutes / swaps y and z directions.
        (Obsolete, use swap_yz)
        """
        self.swap_yz()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def swap_xy(self):
        """
        Swaps x and y axes.
        """
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
        self.val = self.val.swapaxes(1, 2)
        self.ny, self.nz = self.nz, self.ny
        self.sy, self.sz = self.sz, self.sy
        self.oy, self.oz = self.oz, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_xzy(self):
        """
        Apply transposition, send original x, y, z axes to x, z, y axes.
        Equivalent to swap_yz.
        """
        self.val = self.val.transpose((0, 2, 1, 3))
        self.nx, self.ny, self.nz = self.nx, self.nz, self.ny
        self.sx, self.sy, self.sz = self.sx, self.sz, self.sy
        self.ox, self.oy, self.oz = self.ox, self.oz, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_yxz(self):
        """
        Apply transposition, send original x, y, z axes to y, x, z axes.
        Equivalent to swap_xy.
        """
        self.val = self.val.transpose((0, 1, 3, 2))
        self.nx, self.ny, self.nz = self.ny, self.nx, self.nz
        self.sx, self.sy, self.sz = self.sy, self.sx, self.sz
        self.ox, self.oy, self.oz = self.oy, self.ox, self.oz
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_yzx(self):
        """
        Apply transposition, send original x, y, z axes to y, z, x axes.
        """
        self.val = self.val.transpose((0, 2, 3, 1))
        self.nx, self.ny, self.nz = self.nz, self.nx, self.ny
        self.sx, self.sy, self.sz = self.sz, self.sx, self.sy
        self.ox, self.oy, self.oz = self.oz, self.ox, self.oy
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_zxy(self):
        """
        Apply transposition, send original x, y, z axes to z, x, y axes.
        """
        self.val = self.val.transpose((0, 3, 1, 2))
        self.nx, self.ny, self.nz = self.ny, self.nz, self.nx
        self.sx, self.sy, self.sz = self.sy, self.sz, self.sx
        self.ox, self.oy, self.oz = self.oy, self.oz, self.ox
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def transpose_xyz_to_zyx(self):
        """
        Apply transposition, send original x, y, z axes to z, y, x axes.
        Equivalent to swap_xz.
        """
        self.val = self.val.transpose((0, 3, 2, 1))
        self.nx, self.ny, self.nz = self.nz, self.ny, self.nx
        self.sx, self.sy, self.sz = self.sz, self.sy, self.sx
        self.ox, self.oy, self.oz = self.oz, self.oy, self.ox
    # ------------------------------------------------------------------------

    def nxyzv(self):
        return self.nx * self.ny * self.nz * self.nv

    def nxyz(self):
        return self.nx * self.ny * self.nz

    def nxy(self):
        return self.nx * self.ny

    def nxz(self):
        return self.nx * self.nz

    def nyz(self):
        return self.ny * self.nz

    def xmin(self):
        return self.ox

    def ymin(self):
        return self.oy

    def zmin(self):
        return self.oz

    def xmax(self):
        return self.ox + self.nx * self.sx

    def ymax(self):
        return self.oy + self.ny * self.sy

    def zmax(self):
        return self.oz + self.nz * self.sz

    def x(self):
        """
        Returns 1-dimensional array of x coordinates (array of shape (self.nx,)).
        """
        return self.ox + 0.5 * self.sx + self.sx * np.arange(self.nx)

    def y(self):
        """
        Returns 1-dimensional array of y coordinates (array of shape (self.ny,)).
        """
        return self.oy + 0.5 * self.sy + self.sy * np.arange(self.ny)

    def z(self):
        """
        Returns 1-dimensional array of z coordinates (array of shape (self.nz,)).
        """
        return self.oz + 0.5 * self.sz + self.sz * np.arange(self.nz)

    def xx(self):
        """
        Returns mesh of x coordinates:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the x coordinates of each grid cell.
        """
        return np.tile(self.ox + 0.5 * self.sx + self.sx * np.arange(self.nx), self.ny*self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return xx

    def yy(self):
        """
        Returns mesh of y coordinates:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the y coordinates of each grid cell.
        """
        return np.tile(np.repeat(self.oy + 0.5 * self.sy + self.sy * np.arange(self.ny), self.nx), self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return yy

    def zz(self):
        """
        Returns mesh of z coordinates:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the z coordinates of each grid cell.
        """
        return np.repeat(self.oz + 0.5 * self.sz + self.sz * np.arange(self.nz), self.nx*self.ny).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.z(), im.y(), im.x(), indexing='ij')
        # return zz

    def ix(self):
        """
        Returns 1-dimensional array of x index (array of shape (self.nx,)).
        """
        return np.arange(self.nx)

    def iy(self):
        """
        Returns 1-dimensional array of y index (array of shape (self.ny,)).
        """
        return np.arange(self.ny)

    def iz(self):
        """
        Returns 1-dimensional array of z index (array of shape (self.nz,)).
        """
        return np.arange(self.nz)

    def ixx(self):
        """
        Returns mesh of x indexes:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the x indexes of each grid cell.
        """
        return np.tile(np.arange(self.nx), self.ny*self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return xx

    def iyy(self):
        """
        Returns mesh of y indexes:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the y indexes of each grid cell.
        """
        return np.tile(np.repeat(np.arange(self.ny), self.nx), self.nz).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return yy

    def izz(self):
        """
        Returns mesh of z indexes:
            3-dimensional array of shape (self.nz, self.ny, self.nx)
            with the z indexes of each grid cell.
        """
        return np.repeat(np.arange(self.nz), self.nx*self.ny).reshape(self.nz, self.ny, self.nx)
        # equiv:
        # zz, yy, xx = np.meshgrid(im.iz(), im.iy(), im.ix(), indexing='ij')
        # return zz

    def vmin(self):
        return np.nanmin(self.val.reshape(self.nv,self.nxyz()),axis=1)

    def vmax(self):
        return np.nanmax(self.val.reshape(self.nv,self.nxyz()),axis=1)
# ============================================================================

# ============================================================================
class PointSet(object):
    """
    Defines a point set:
        npt:     (int) size of the point set (number of points)
        nv:      (int) number of variables (including x, y, z coordinates)
        val:     ((nv,npt) array) attribute(s) / variable(s) values
        varname: (list of string (or string)) variable names
        name:    (string) name of the point set
    """

    def __init__(self,
                 npt=0,
                 nv=0, val=np.nan, varname=None,
                 name=""):
        """
        Inits function for the class:

        :param val: (int/float or tuple/list/ndarray) value(s) of the new
                        variable:
                        if type is int/float: constant variable
                        if tuple/list/ndarray: must contain npt values
        """

        fname = 'PointSet'

        self.npt = int(npt)
        self.nv = int(nv)

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(npt*nv)
        elif valarr.size != npt*nv:
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        self.val = valarr.reshape(nv, npt)

        if varname is None:
            self.varname = []

            if nv > 0:
                self.varname.append("X")

            if nv > 1:
                self.varname.append("Y")

            if nv > 2:
                self.varname.append("Z")

            if nv > 3:
                for i in range(3,nv):
                    self.varname.append("V{:d}".format(i-3))

        else:
            varname = list(np.asarray(varname).reshape(-1))
            if len(varname) != nv:
                print(f'ERROR ({fname}): varname has not an acceptable size')
                return None

            self.varname = list(np.asarray(varname).reshape(-1))

        self.name = name

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
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
        Sets default variable names: 'X', 'Y', 'Z', 'V0', 'V1', ...
        """

        self.varname = []

        if self.nv > 0:
            self.varname.append("X")

        if self.nv > 1:
            self.varname.append("Y")

        if self.nv > 2:
            self.varname.append("Z")

        if self.nv > 3:
            for i in range(3,self.nv):
                self.varname.append("V{:d}".format(i-3))
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_varname(self, varname=None, ind=-1):
        """
        Sets name of the variable of the given index (if varname is None:
        'V' appended by the variable index is used as varname).
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        if varname is None:
            varname = "V{:d}".format(ii)
        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, val=np.nan, varname=None, ind=0):
        """
        Inserts one or several variable(s) at a given index.

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: one constant variable is
                                inserted
                            if tuple/list/ndarray: its size must be a multiple
                                of self.npt
        :param varname: (string, or tuple/list/1-d array of strings or None)
                            name(s) of the new variable(s), if not given (None),
                            default variable names are set ("V<num>", where
                            <num> starts from the number of variables before
                            inserting)
        :param ind:     (int) index where the new variable(s) is (are) inserted
        """

        fname = 'insert_var'

        # Check / set ind
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            print("Nothing is done! (invalid index)")
            return None

        # Check val, set valarr (array of values)
        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size % self.npt != 0:
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        m = valarr.size // self.npt # number of variable to be inserted

        # Check / set varname
        if varname is not None:
            if isinstance(varname, str):
                varname = [varname]
            elif (not isinstance(varname, tuple) and not isinstance(varname, list) and not (isinstance(varname, np.ndarray) and im_list.ndim==1)) or len(varname)!=m:
                print(f'ERROR ({fname}): varname does not have an acceptable size')
                return None
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
    def append_var(self, val=np.nan, varname=None):
        """
        Appends one or several variable(s).

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: one constant variable is
                                inserted
                            if tuple/list/ndarray: its size must be a multiple
                                of self.nx*self.ny*self.nz
        :param varname: (string, or tuple/list/1-d array of strings or None)
                            name(s) of the new variable(s), if not given (None),
                            default variable names are set ("V<num>", where
                            <num> starts from the number of variables before
                            appending)
        """

        self.insert_var(val=val, varname=varname, ind=self.nv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=None, indlist=None):
        """
        Removes variable(s) (of given index-es).

        :param ind:     (int or list of ints) index or list of index-es of the
                            variable(s) to be removed

        :param indlist: used for ind if ind is not given (None)
                            (obsolete, kept for compatibility with older
                                versions)
        """

        if ind is None:
            ind = indList
            if ind is None:
                print("Nothing is done! (no index given)")
                return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            print("Nothing is done! (invalid index)")
            return

        ind = np.setdiff1d(np.arange(self.nv), ind)

        self.extract_var(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """
        Removes all variables.
        """

        # Update val array
        self.val = np.zeros((0, self.npt))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, val=np.nan, varname=None, ind=-1):
        """
        Sets one variable (of given index).

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain npt values
        :param varname: (string or None) name of the new variable
        :param ind:     (int) index where the variable to be set
        """

        fname = 'set_var'

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            print(f'ERROR ({fname}): val does not have an acceptable size')
            return None

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.npt)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, ind=None, indlist=None):
        """
        Extracts variable(s) (of given index-es).
        (May be used for reordering / duplicating variables.)

        :param ind:     (int or list of ints) index or list of index-es of the
                            variable(s) to be extracted (kept)

        :param indlist: used for ind if ind is not given (None)
                            (obsolete, kept for compatibility with older
                                versions)
        """

        if ind is None:
            ind = indList
            if ind is None:
                print("Nothing is done! (no index given)")
                return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allvar()
            return

        ind[ind<0] = self.nv + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.nv)):
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        self.val = self.val[ind,...]

        # Update varname list
        self.varname = [self.varname[i] for i in ind]

        # Update nv
        self.nv = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_point(self, ind=None):
        """
        Removes point(s) (of given index-es).

        :param ind:     (int or list of ints) index or list of index-es of the
                            point(s) to be removed
        """

        if ind is None:
            print("Nothing is done! (no index given)")
            return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            return

        ind[ind<0] = self.npt + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.npt)):
            print("Nothing is done! (invalid index)")
            return

        ind = np.setdiff1d(np.arange(self.npt), ind)

        self.extract_point(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allpoint(self):
        """
        Removes all points.
        """

        # Update val array
        self.val = np.zeros((self.nv, 0))

        # Update npt
        self.npt = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_uninformed_point(self):
        """
        Removes point(s) where all variables are undefined (nan).
        """

        # Get index of variables that are not coordinates
        ind = np.where([not (self.varname[i] in ('x', 'X', 'y', 'Y', 'z', 'Z')) for i in range(self.nv)])

        # Remove uninformed points
        self.val = self.val[:, ~np.all(np.isnan(self.val[ind]), axis=0)]

        # Update npt
        self.npt = self.val.shape[1]
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_point(self, ind=None):
        """
        Extracts point(s) (of given index-es).

        :param ind:     (int or list of ints) index or list of index-es of the
                            point(s) to be extracted (kept)
        """

        if ind is None:
            print("Nothing is done! (no index given)")
            return

        ind = np.atleast_1d(ind).reshape(-1)
        if ind.size == 0:
            self.remove_allpt()
            return

        ind[ind<0] = self.npt + ind[ind<0] # deal with negative index-es
        if np.any((ind < 0, ind >= self.npt)):
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        self.val = self.val[:, ind]

        # Update npt
        self.npt = len(ind)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique_one_var(self, ind=0, ignore_missing_value=True):
        """
        Gets unique values of one variable (of given index).

        :param ind: (int) index of the variable
        :param ignore_missing_value:
                    (bool) if True: missing value (nan), if present, are ignored;
                        if False: missing value (nan), if present, is retrieved in
                        output

        :return:    (1-dimensional array) unique values of the variable
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        uval = np.unique(self.val[ii])

        if ignore_missing_value:
            uval = uval[~np.isnan(uval)]

        return uval
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True, ignore_missing_value=True):
        """
        Gets proportions (density or count) of unique values of one
        variable (of given index).

        :param ind:     (int) index of the variable
        :param density: (bool) computes densities if True and counts otherwise
        :param ignore_missing_value:
                        (bool) if True: missing value (nan), if present, are
                            ignored; if False: missing value (nan), if present,
                            is taken into account

        :return out:    (2-tuple of 1-dimensional array)
                            out[0]: (1-dimensional array) unique values of
                                    the variable
                            out[1]: (1-dimensional array) densities or counts of
                                    the unique values
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return None

        uv, cv = np.unique(self.val[ii], return_counts=True)

        if ignore_missing_value:
            ind_known = ~np.isnan(uv)
            uv = uv[ind_known]
            cv = cv[ind_known]

        if density:
            cv = cv / np.sum(cv)

        return (uv, cv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def to_dict(self):
        """
        Returns PointSet as a dictionary.
        """
        return {name: values for name, values in zip(self.varname, self.val)}
    # ------------------------------------------------------------------------

    def x(self):
        return self.val[0]

    def y(self):
        return self.val[1]

    def z(self):
        return self.val[2]

    def xmin(self):
        return np.min(self.val[0])

    def ymin(self):
        return np.min(self.val[1])

    def zmin(self):
        return np.min(self.val[2])

    def xmax(self):
        return np.max(self.val[0])

    def ymax(self):
        return np.max(self.val[1])

    def zmax(self):
        return np.max(self.val[2])
# ============================================================================

# ----------------------------------------------------------------------------
def copyImg(im, varInd=None, varIndList=None):
    """
    Copies an image (Img class), with all variables or a subset of variables.

    :param im:          (Img class) input image
    :param varInd:      (sequence of ints or int or None) index-es of
                            the variables to be copied, use varInd=[] to copy
                            only the grid geometry
                            (default None: all variables are copied)"
    :param varIndList:  used for varInd if varInd is not given (None)
                            (obsolete, kept for compatibility with older
                            versions)

    :return:   (Img class) a copy of the input image (not a reference to)
    """

    fname = 'copyImg'

    if varInd is None:
        varInd = varIndList

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if varInd.size == 0:
            # empty list of variable
            imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                        sx=im.sx, sy=im.sy, sz=im.sz,
                        ox=im.ox, oy=im.oy, oz=im.oz,
                        nv=0, name=im.name)
        else:
            # Check if each index is valid
            if np.sum([iv in range(im.nv) for iv in varInd]) != len(varInd):
                print(f'ERROR ({fname}): invalid index-es')
                return None
            imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                        sx=im.sx, sy=im.sy, sz=im.sz,
                        ox=im.ox, oy=im.oy, oz=im.oz,
                        nv=len(varInd), val=im.val[varInd], varname=[im.varname[i] for i in varInd],
                        name=im.name)
            # imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
            #             sx=im.sx, sy=im.sy, sz=im.sz,
            #             ox=im.ox, oy=im.oy, oz=im.oz,
            #             nv=len(varInd),
            #             name=im.name)
            # for i, iv in enumerate(varInd):
            #     imOut.set_var(val=im.val[iv,...], varname=im.varname[iv], ind=i)
    else:
        # Copy all variables
        imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                    sx=im.sx, sy=im.sy, sz=im.sz,
                    ox=im.ox, oy=im.oy, oz=im.oz,
                    nv=im.nv, val=np.copy(im.val), varname=list(np.copy(np.asarray(im.varname))),
                    name=im.name)

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def copyPointSet(ps, varInd=None, varIndList=None):
    """
    Copies point set, with all variables or a subset of variables.

    :param ps:          (PointSet class) input point set
    :param varInd:      (sequence of ints or int or None) index-es of the
                            variables to be copied (default None: all variables)"
    :param varIndList:  used for varInd if varInd is not given (None)
                            (obsolete, kept for compatibility with older
                            versions)

    :return:    (PointSet class) a copy of the input point set (not a reference
                    to)
    """

    fname = 'copyPointSet'

    if varInd is None:
        varInd = varIndList

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        # Check if each index is valid
        if np.sum([iv in range(ps.nv) for iv in varInd]) != len(varInd):
            print(f'ERROR ({fname}): invalid index-es')
            return None
        psOut = PointSet(npt=ps.npt,
                         nv=len(varInd), val=ps.val[varInd], varname=[ps.varname[i] for i in varInd],
                         name=ps.name)
    else:
        # Copy all variables
        psOut = PointSet(npt=ps.npt,
                         nv=ps.nv, val=np.copy(ps.val), varname=list(np.copy(np.asarray(ps.varname))),
                         name=ps.name)

    return psOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageToPointSet(im, remove_uninformed_cell=True):
    """
    Returns a point set corresponding to the input image.

    Note that any image cell with no value (i.e. all variables are missing (nan))
    is not considered in the output point set.

    :param im:  (Img class) input image
    :param remove_uninformed_cell:
                (bool) if True, any image cell with no value (i.e. all variables
                    are missing (nan)) is not considered in the output point set;
                    if False, every image cell are considered in the output point
                    set

    :return ps: (PointSet class) point set corresponding to the input image
    """

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
def pointSetToImage(ps,
                    nx=None, ny=None, nz=None,
                    sx=None, sy=None, sz=None,
                    ox=None, oy=None, oz=None,
                    nx_max=10000, ny_max=10000, nz_max=10000, nxyz_max=10000000,
                    sx_min=1.e-6, sy_min=1.e-6, sz_min=1.e-6,
                    job=0):
    """
    Returns an image corresponding to the input point set.
    The output image grid geometry is defined according to the input parameters
    (nx, ny, nz, sx, sy, sz, ox, oy, oz). Parameters not given (None) are
    automatically computed such that, if possible, the grid covers all points of
    the input point set, with cell size such that only one point is in a same
    cell.

    Note: the last point is selected if more than one point fall in a same cell.

    :param ps:  (PointSet class) input point set, with x, y, z-coordinates as
                    first three variable
    :param nx, ny, nz:
                (int or None) number of grid cells along each axis
    :param sx, sy, sz:
                (float or None) cell size along each axis
    :param ox, oy, oz:
                (float or None) origin of the grid (bottom-lower-left corner)
    :param nx_max, ny_max, nz_max:
                (int) maximal values for nx, ny, nz
    :param nxyz_max:
                (int) maximal value for the product nx*ny*nz
    :param sx_min, sy_min, sz_min:
                (float) minimal values for sx, sy, sz
    :param job: (int) defines some behaviour:
                    - if 0: an error occurs if one data is located outside of
                        the image grid, otherwise all data are integrated in the
                        image
                    - if 1: data located outside of the image grid are ignored
                        (no error occurs), and all data located within the image
                        grid are integrated in the image

    :return im: (Img class) image corresponding to the input point set and grid
    """

    fname = 'pointSetToImage'

    if ps.nv < 3:
        print(f'ERROR ({fname}): invalid number of variable (should be > 3)')
        return None

    if ps.varname[0].lower() != 'x' or ps.varname[1].lower() != 'y' or ps.varname[2].lower() != 'z':
        print(f'ERROR ({fname}): invalid variable: 3 first ones must be x, y, z coordinates')
        return None

    if (nx is None or ny is None or nz is None \
    or sx is None or sy is None or sz is None \
    or ox is None or oy is None or oz is None) \
    and ps.npt == 0:
        print(f'ERROR ({fname}): number of point is 0, unable to compute grid geometry')
        return None

    # Compute cell size (if not given)
    if sx is None:
        t = np.unique(ps.x())
        if t.size > 1:
            sx = max(np.min(np.diff(t)), sx_min)
        else:
            sx = 1.0
    if sy is None:
        t = np.unique(ps.y())
        if t.size > 1:
            sy = max(np.min(np.diff(t)), sy_min)
        else:
            sy = 1.0
    if sz is None:
        t = np.unique(ps.z())
        if t.size > 1:
            sz = max(np.min(np.diff(t)), sz_min)
        else:
            sz = 1.0

    # Compute origin (if not given)
    if ox is None:
        ox = ps.x().min() - 0.5*sx
    if oy is None:
        oy = ps.y().min() - 0.5*sy
    if oz is None:
        oz = ps.z().min() - 0.5*sz

    # Compute dimension, i.e. number of cells (if not given)
    if nx is None:
        nx = min(int(np.ceil((ps.x().max() - ox)/sx)), nx_max)
    if ny is None:
        ny = min(int(np.ceil((ps.y().max() - oy)/sy)), ny_max)
    if nz is None:
        nz = min(int(np.ceil((ps.z().max() - oz)/sz)), nz_max)

    # Initialize image
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=ps.nv-3, val=np.nan,
             varname=[ps.varname[3+i] for i in range(ps.nv-3)])

    # Get index of point in the image
    xmin, xmax = im.xmin(), im.xmax()
    ymin, ymax = im.ymin(), im.ymax()
    zmin, zmax = im.zmin(), im.zmax()
    ix = np.array(np.floor((ps.val[0]-xmin)/sx),dtype=int)
    iy = np.array(np.floor((ps.val[1]-ymin)/sy),dtype=int)
    iz = np.array(np.floor((ps.val[2]-zmin)/sz),dtype=int)
    # ix = [np.floor((x-xmin)/sx + 0.5) for x in ps.val[0]]
    # iy = [np.floor((y-ymin)/sy + 0.5) for y in ps.val[1]]
    # iz = [np.floor((z-zmin)/sz + 0.5) for z in ps.val[2]]
    for i in range(ps.npt):
        if ix[i] == nx:
            if (ps.val[0,i]-xmin)/sx - nx < 1.e-10:
                ix[i] = nx-1

        if iy[i] == ny:
            if (ps.val[1,i]-ymin)/sy - ny < 1.e-10:
                iy[i] = ny-1

        if iz[i] == nz:
            if (ps.val[2,i]-zmin)/sz - nz < 1.e-10:
                iz[i] = nz-1

    # Check which index is out of the image grid
    iout = np.any(np.array((ix < 0, ix >= nx, iy < 0, iy >= ny, iz < 0, iz >= nz)), axis=0)

    if not job and np.sum(iout) > 0:
        print(f'ERROR ({fname}): point out of the image grid!')
        return None

    # Set values in the image (last point is selected if more than one in a cell)
    for i in range(ps.npt): # ps.npt is equal to iout.size
        if not iout[i]:
            # if not np.isnan(im.val[0, iz[i], iy[i], ix[i]]:
            #     print(f'WARNING ({fname}): more than one point in the same cell!')
            im.val[:,iz[i], iy[i], ix[i]] = ps.val[3:ps.nv,i]

    if np.sum(~np.isnan(im.val[0])) != ps.npt:
        print(f'WARNING ({fname}): more than one point in the same cell!')

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def isImageDimensionEqual(im1, im2):
    """
    Checks if grid dimensions of two images are equal.
    """

    return im1.nx == im2.nx and im1.ny == im2.ny and im1.nz == im2.nz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def isImageEqual(im1, im2):
    """
    Checks if two images are equal (dimension, spacing, origin, variables).
    """

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
    Checks if two point sets are equal (npt, nv, variable values).
    """

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
def indicatorImage(im, ind=0, categ=None):
    """
    Retrieve the image with the indicator variable of each category in the list
    of categories 'categ', from the variable of index 'ind' in the input image
    'im'.

    :param im:      (Img class) input image
    :param ind:     (int) index of the variable in the input image for which
                        the indicator variable(s) are computed
    :param categ:   (sequence of values or float (or int) or None)
                        list of category value: one indicator variable per value
                        in that list is computed for the variable of index 'ind'
                        in the input image; if None (default), categ is set to
                        the list of all distinct values (in increasing order)
                        taken by the variable of index 'ind' in the input image

    :return:    (Img class) output image with indicator variable(s) (as many
                    variable(s) as number of category values given by 'categ')
    """

    fname = 'indicatorImage'

    # Check (set) ind
    if ind < 0:
        ind = im.nv + ind

    if ind < 0 or ind >= im.nv:
        print(f'ERROR ({fname}): invalid index')
        return None

    # Set categ if not given (None)
    if categ is None:
        categ = im.get_unique_one_var(ind=ind)

    ncateg = len(categ)

    # Initialize an image with ncateg variables
    im_out = Img(
        nx=im.nx, ny=im.ny, nz=im.nz,
        sx=im.sx, sy=im.sy, sz=im.sz,
        ox=im.ox, oy=im.oy, oz=im.oz,
        nv=ncateg, varname=[f'{im.varname[ind]}_ind{i:03d}' for i in range(ncateg)])

    # Compute each indicator variable
    for i, v in enumerate(categ):
        val = 1.*(im.val[ind]==v)
        val[np.where(np.isnan(im.val[ind]))] = np.nan
        im_out.val[i,...] = val

    return im_out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def gatherImages(imlist, varInd=None, keep_varname=False, rem_var_from_source=False, treat_image_one_by_one=False):
    """
    Gathers images.

    :param imlist:  (list) images to be gathered, they should have the same grid
                        dimensions
    :param varInd:  (sequence of ints or int or None) index-es of the variables
                        of each image from imlist to be retrieved
                        - if None (default): all variables of each image from
                            'imlist' are stored in the output image
                        - else: only the variables of index in varInd of each
                            image from imlist is stored in the output image
    :param keep_varname:
                    (bool) if True, name of the variables are kept from the
                       source, else (False), default variable names are set

    :param rem_var_from_source:
                    (bool) if True, gathered variables are removed from the
                        source (list of input images) (this allows to save
                        memory)

    :param treat_image_one_by_one:
                    (bool) note: if rem_var_from_source is set to False, then
                        treat_image_one_by_one is ignored (as it was set to
                        False): there is no need to deal with images one by one
                        - treat_image_one_by_one=True: images of the input list
                            are treated one by one, i.e. the variables to be
                            gathered of each image are inserted in the output
                            image and removed from the source (slower, may save
                            memory)
                        - treat_image_one_by_one=False: all images of the input
                            list are treated at once, i.e. variables to be
                            gathered of all images are inserted in the output
                            image at once (faster)

    :return im: (Img class) output image containing variables to be gathered of
                    images in imlist
    """

    fname = 'gatherImages'

    if len(imlist) == 0:
        return None

    for i in range(1,len(imlist)):
        if not isImageDimensionEqual(imlist[0], imlist[i]):
            print(f'ERROR ({fname}): grid dimensions differ, nothing done!')
            return None

    if varInd is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        if np.sum([iv in range(im.nv) for im in imlist for iv in varInd]) != len(imlist)*len(varInd):
            print(f'ERROR ({fname}): invalid index-es')
            return None

    varname = None # default
    if keep_varname:
        if varInd is not None:
            varname = [im.varname[iv] for im in imlist for iv in varInd]
        else:
            varname = [im.varname[iv] for im in imlist for iv in range(im.nv)]

    if rem_var_from_source:
        # remove variable from source
        if treat_image_one_by_one:
            # treat images one by one
            val = np.empty(shape=(0, imlist[0].nz, imlist[0].ny, imlist[0].nx))
            if varInd is not None:
                ind = np.sort(np.unique(varInd))[::-1] # unique index in decreasing order (for removing variable...)
                for im in imlist:
                    val = np.concatenate((val, im.val[varInd]), 0)
                    for iv in ind:
                        im.remove_var(iv)
            else:
                for im in imlist:
                    val = np.concatenate((val, im.val), 0)
                    im.remove_allvar()
        else:
            # treat all images at once
            if varInd is not None:
                val = np.concatenate([im.val[varInd] for im in imlist], 0)
                ind = np.sort(np.unique(varInd))[::-1] # unique index in decreasing order (for removing variable...)
                for im in imlist:
                    for iv in ind:
                        im.remove_var(iv)
            else:
                val = np.concatenate([im.val for im in imlist], 0)
                for im in imlist:
                    im.remove_allvar()
    else:
        # not remove variable from source
        # ignore treat_image_one_by_one (as it was False)
        # treat_image_one_by_one = False # changed if needed: no need to treat images one by one...
        #
        # treat all images at once
        if varInd is not None:
            val = np.concatenate([im.val[varInd] for im in imlist], 0)
        else:
            val = np.concatenate([im.val for im in imlist], 0)

    im = Img(
            nx=imlist[0].nx, ny=imlist[0].ny, nz=imlist[0].nz,
            sx=imlist[0].sx, sy=imlist[0].sy, sz=imlist[0].sz,
            ox=imlist[0].ox, oy=imlist[0].oy, oz=imlist[0].oz,
            nv=val.shape[0], val=val, varname=varname)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageContStat(im, op='mean', **kwargs):
    """
    Computes "pixel-wise" statistics over every variable of an image.

    :param im:      (Img class) input image
    :param op:      (string) statistic operator, can be:
                        'max': max
                        'mean': mean
                        'min': min
                        'std': standard deviation
                        'var': variance
                        'quantile': quantile
                                    this operator requires the keyword argument
                                    q=<sequence of quantile to compute>
    :param kwargs:  additional key word arguments passed to np.<op> function,
                        typically: ddof=1 if op is 'std' or 'var'

    :return:    (Img class) image with same grid as the input image and one
                    variable being the pixel-wise statistics according to 'op'
                    over every variable of the input image
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
            print(f"ERROR ({fname}): keyword argument 'q' required for op='quantile', nothing done!")
            return None
        varname = [op + '_' + str(v) for v in kwargs['q']]
    else:
        print(f"ERROR ({fname}): unkown operation '{op}', nothing done!")
        return None

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=0, val=0.0)

    vv = func(im.val.reshape(im.nv,-1), axis=0, **kwargs)
    vv = vv.reshape(-1, im.nxyz())
    for v, name in zip(vv, varname):
        imOut.append_var(v, varname=name)

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageListContStat(im_list, op='mean', ind=0, **kwargs):
    """
    Computes "pixel-wise" statistics over one variable of a list of images.

    :param im_list: (list or 1d-array of Img (class)) list of input images
                        defined on the same grid and having the same variables
                        (e.g. realizations)
    :param op:      (string) statistic operator, can be:
                        'max': max
                        'mean': mean
                        'min': min
                        'std': standard deviation
                        'var': variance
                        'quantile': quantile
                                    this operator requires the keyword argument
                                    q=<sequence of quantile to compute>
    :param ind:     (int) index of the variable in the input images for which
                        the statistics are computed
    :param kwargs:  additional key word arguments passed to np.<op> function,
                        typically: ddof=1 if op is 'std' or 'var'

    :return:    (Img class) image with same grid as the input images and one
                    variable being the pixel-wise statistics according to 'op'
                    over the variable of index 'ind' of the input images
    """

    fname = 'imageListContStat'

    # Check input images
    if not isinstance(im_list, list) and not (isinstance(im_list, np.ndarray) and im_list.ndim==1):
        print(f'ERROR ({fname}): first argument must be a list (or a 1d-array) of images')
        return None

    if len(im_list) == 0:
        return None

    im0 = im_list[0]
    for im in im_list[1:]:
        if im.val.shape != im0.val.shape:
            print(f'ERROR ({fname}): images in list of incompatible size')
            return None

    # Check (set) ind
    if ind < 0:
        ind = im0.nv + ind

    if ind < 0 or ind >= im0.nv:
        print(f'ERROR ({fname}): invalid index')
        return None

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
            print(f"ERROR ({fname}): keyword argument 'q' required for op='quantile', nothing done!")
            return None
        varname = [op + '_' + str(v) for v in kwargs['q']]
    else:
        print(f"ERROR ({fname}): unkown operation '{op}', nothing done!")
        return None

    imOut = Img(nx=im0.nx, ny=im0.ny, nz=im0.nz,
                sx=im0.sx, sy=im0.sy, sz=im0.sz,
                ox=im0.ox, oy=im0.oy, oz=im0.oz,
                nv=0, val=0.0)

    vv = func(np.asarray([im.val[ind] for im in im_list]).reshape(len(im_list),-1), axis=0, **kwargs)
    vv = vv.reshape(-1, im.nxyz())
    for v, name in zip(vv, varname):
        imOut.append_var(v, varname=name)

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageCategProp(im, categ):
    """
    Computes "pixel-wise" proportions of given categories over every
    variable of an image.

    :param im:      (Img class) input image
    :param categ:   (sequence) list of value(s) for which the proportions
                        are computed

    :return:    (Img class) image with same grid as the input image and as many
                    variable(s) as given by 'categ', being the pixel-wise
                    proportions of each category in 'categ', over every variable
                    of the input image
    """

    # Array of categories
    categarr = np.array(categ,dtype=float).reshape(-1)

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=0, val=0.0)

    for i, code in enumerate(categarr):
        x = 1.0*(im.val.reshape(im.nv,-1) == code)
        np.putmask(x, np.isnan(im.val.reshape(im.nv,-1)), np.nan)
        imOut.append_var(np.mean(x, axis=0), varname=f'prop{i}')

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageListCategProp(im_list, categ, ind=0):
    """
    Computes "pixel-wise" proportions of given categories over one variable of a
    list of images.

    :param im_list: (list or 1d-array of Img (class)) list of input images
                        defined on the same grid and having the same variables
                        (e.g. realizations)
    :param categ:   (sequence) list of value(s) for which the proportions
                        are computed
    :param ind:     (int) index of the variable in the input images for which
                        the statistics are computed

    :return:    (Img class) image with same grid as the input images and as many
                    variable(s) as given by 'categ', being the pixel-wise
                    proportions of each category in 'categ', over the variable
                    of index 'ind' of the input images
    """

    fname = 'imageListCategProp'

    # Check input images
    if not isinstance(im_list, list) and not (isinstance(im_list, np.ndarray) and im_list.ndim==1):
        print(f'ERROR ({fname}): first argument must be a list (or a 1d-array) of images')
        return None

    if len(im_list) == 0:
        return None

    im0 = im_list[0]
    for im in im_list[1:]:
        if im.val.shape != im0.val.shape:
            print(f'ERROR ({fname}): images in list of incompatible size')
            return None

    # Check (set) ind
    if ind < 0:
        ind = im0.nv + ind

    if ind < 0 or ind >= im0.nv:
        print(f'ERROR ({fname}): invalid index')
        return None

    # Array of categories
    categarr = np.array(categ,dtype=float).reshape(-1)

    imOut = Img(nx=im0.nx, ny=im0.ny, nz=im0.nz,
                sx=im0.sx, sy=im0.sy, sz=im0.sz,
                ox=im0.ox, oy=im0.oy, oz=im0.oz,
                nv=0, val=0.0)

    v = np.asarray([im.val[ind] for im in im_list]).reshape(len(im_list),-1)
    for i, code in enumerate(categarr):
        x = 1.0*(v == code)
        np.putmask(x, np.isnan(v), np.nan)
        imOut.append_var(np.mean(x, axis=0), varname=f'prop{i}')

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageEntropy(im, varInd=None, varIndList=None):
    """
    Computes "pixel-wise" entropy for proprotions given as variables in an
    image.

    :param im:          (Img class) input image
    :param varInd:      (sequence of ints or None) index-es of the variables
                            to take into account (default None: all variables),
                            (length of varInd should be at least 2)
    :param varIndList:  used for varInd if varInd is not given (None)
                            (obsolete, kept for compatibility with older
                            versions)

    :return:    (Img class) an image with one variable containing the entropy
                    for the variable given in input, at pixel i, it is defined
                    as:
                        Ent(i) = - sum_{v} p_v(i) * log_n(p(v(i)))
                    where v loops on each variable and n is the number of
                    variables. Note that sum_{v} p(v(i)) should be equal to 1
    """

    fname = 'imageEntropy'

    if varInd is None:
        varInd = varIndList

    if varIndList is not None:
        varInd = np.atleast_1d(varInd).reshape(-1)
        # Check if each index is valid
        if np.sum([iv in range(im.nv) for iv in varInd]) != len(varInd):
            print(f'ERROR ({fname}): invalid index-es')
            return None
    else:
        varInd = range(im.nv)

    if len(varInd) < 2:
        print(f'ERROR ({fname}): at least 2 indexes should be given')
        return None

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=1, val=np.nan,
                name=im.name)

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
                    imOut.val[0][iz][iy][ix] = t*e

    return imOut
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pointToGridIndex(x, y, z, sx=1.0, sy=1.0, sz=1.0, ox=0.0, oy=0.0, oz=0.0):
    """
    Convert real point coordinates to index grid.

    :param x, y, z:     (float, or 1-d array of floats) coordinates of point(s)
    :param sx, sy, sz:  (float) cell size along each axis
    :param ox, oy, oz:  (float) origin of the grid (bottom-lower-left corner)

    :return (ix, iy, iz):
                        (3-tuple of floats or 1-d of floats) grid node index
                            in x-, y-, z-axis direction respectively for each
                            point given in input
                            Warning: no check if the node(s) is within the grid
    """
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
def gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz):
    """
    Convert a grid index (3 indices) into a single grid index.

    :param ix, iy, iz:  (int) grid index in x-, y-, z-axis direction
    :param nx, ny, nz:  (int) number of grid cells along each axis

    :return i:  (int) single grid index
                    Note: ix, iy, iz can be ndarray of same shape, then i is a
                    ndarray of that shape
    """
    return ix + nx * (iy + ny * iz)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def singleGridIndexToGridIndex(i, nx, ny, nz):
    """
    Convert a single into a grid index (3 indices).

    :param i:           (int) single grid index
    :param nx, ny, nz:  (int) number of grid cells along each axis

    :return (ix, iy, iz):
                        (3-tuple) grid index in x-, y-, z-axis direction
                            Note: i can be a ndarray, then ix, iy, iz in output
                            are ndarray (of same shape)
    """
    nxy = nx*ny
    iz = i//nxy
    j = i%nxy
    iy = j//nx
    ix = j%nx

    return ix, iy, iz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def aggregateDataPointsInGrid(x, y, z, v, nx, ny, nz, sx=1.0, sy=1.0, sz=1.0, ox=0.0, oy=0.0, oz=0.0):
    """
    Remove points out of the grid and aggregate data points falling in a same
    grid cell by taking the mean coordinates and the mean value for each
    variable.

    :param x, y, z:     (1-d array of floats) coordinates of point(s)
    :param v:           (float, or 1-d array of floats, or 2-d array of floats)
                            values attached to point(s), each row of v (if 2d
                            array) corresponds to a same variable
    :param nx, ny, nz:  (int) number of grid cells along each axis
    :param sx, sy, sz:  (float) cell size along each axis
    :param ox, oy, oz:  (float) origin of the grid (bottom-lower-left corner)

    :return (x, y, z, v):
                        (4-tuple): points with aggregated information
                            x, y, z: 1-d arrays of floats
                            v: 1- or 2-d arrays of floats
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    v = np.atleast_2d(v)

    # Keep only the points within the grid
    ind = np.all((x >= ox, x <= ox+sx*nx, y >= oy, y <= oy+sy*ny, z >= oz, z <= oz+sz*nz), axis=0)
    if not np.any(ind):
        # no point in the grid
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        v = np.zeros(shape=np.repeat(0, v.ndim))
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
    ic = ix + nx * (iy + ny * iz) # single-indices

    ic_unique, ic_inv = np.unique(ic, return_inverse=True)
    if len(ic_unique) != len(ic):
        nxy = nx*ny
        ic = ic_unique
        iz = ic//nxy
        j = ic%nxy
        iy = j//nx
        ix = j%nx
        c = np.array([c[ic_inv==j].mean(axis=0) for j in range(len(ic_unique))])
        v = np.array([v[ic_inv==j].mean(axis=0) for j in range(len(ic_unique))])

    x, y, z = c.T # unpack
    v = v.T

    return x, y, z, v
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sampleFromPointSet(point_set, size, seed=None, mask=None):
    """
    Sample random points from PointSet object and return a point set.

    :param point_set:   (PointSet class) point set to sample from
    :param size:        (int) number of points to be sampled
    :param seed:        (int) optional random seed
    :param mask:        (PointSet class) point set of the same size showing where
                            to sample points where mask == 0 will be not taken
                            into account

    :return:    (PointSet class) a point set containing the sample points
    """
    # Initialise the seed; will randomly reseed the generator if None
    np.random.seed(seed)

    if mask is not None:
        indices = np.where(mask.val[3,:] != 0)[0]
    else:
        indices = point_set.npt

    # Sample only some points from the point set
    sampled_indices = np.random.choice(indices, size, replace=False)

    # Return the new object
    return PointSet(npt=size,
            nv=point_set.nv,
            val=point_set.val[:,sampled_indices],
            varname=point_set.varname,
            name=point_set.name)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sampleFromImage(image, size, seed=None, mask=None):
    """
    Samples random points from Img object and returns a point set.

    :param image:   (Img class) image to sample from
    :param size:    (int) number of points to be sampled
    :param seed:    (int) optional random seed
    :param mask:    (Img class) image of the same size indicating where to
                        sample points where mask == 0 will be not taken into
                        account

    :return:    (PointSet class) a point set containing the sample points
    """
    # Create point set from image
    point_set = imageToPointSet(image)
    if mask is not None:
        mask = imageToPointSet(mask)

    return sampleFromPointSet(point_set, size, seed, mask)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def extractRandomPointFromImage(im, npt, seed=None):
    """
    Extracts random points from an image (at center of grid cells) and return
    the corresponding point set.

    :param im:  (Img class) input image
    :param npt: (int) number of points to extract (if greater than the number of
                    image grid cells, 'npt' is set to this latter)
    :seed:      (int) seed number for initializing the random number generator
                    (if not None)

    :return:    (PointSet class) a point set containing the sample points
    """

    fname = 'extractRandomPointFromImage'

    if npt <= 0:
        print(f'ERROR ({fname}): number of points negative or zero (npt={npt}), nothing done!')
        return None

    if npt >= im.nxyz():
        return imageToPointSet(im)

    if seed is not None:
        np.random.seed(seed)

    # Get random single grid indices
    ind_grid = np.random.choice(np.arange(im.nxyz()), size=npt, replace=False)

    # Get grid indices along each axis
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
    ps.set_var(val=x, varname='X', ind=0)
    ps.set_var(val=y, varname='Y', ind=1)
    ps.set_var(val=z, varname='Z', ind=2)

    # Set next variable(s)
    for i in range(im.nv):
        ps.set_var(val=v[:,i], varname=im.varname[i], ind=3+i)

    return ps
# ----------------------------------------------------------------------------

# === Read / Write function below ===

# ----------------------------------------------------------------------------
def readVarsTxt(fname, missing_value=None, delimiter=' ', comments='#', usecols=None):
    """
    Reads variables from a txt file in the following format.

    --- file (ascii) ---
    # commented line ...
    # [...]
    varname[0] varname[1] ... varname[nv-1]
    v[0, 0]    v[0, 1]    ... v[0, nv-1]
    v[1, 0]    v[1, 1]    ... v[1, nv-1]
    ...
    v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param fname:           (string or file handle) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
                                Note: "empty field" after splitting is ignored,
                                then if white space is used as delimiter
                                (default), one or multiple white spaces can be
                                used as the same delimiter
    :param comments:        (string or None) lines starting with that string are
                                treated as comments
    :param usecols:         (tuple or int or None) columns index (first column
                                is index 0) to be read, if None (default) all
                                columns are read

    :return (varname, val): (2-tuple)
                                varname: (list of string) list of variable names,
                                varname[i] being the name of the variable of
                                index i
                                val: (2d array) values of the variables, with
                                val[:,i] the values of variable of index i
    """

    funcname = 'readVarsTxt'

    # Check comments identifier
    if comments is not None and comments == '':
        print(f'ERROR ({funcname}): comments cannot be an empty string, use comments=None to disable comments')
        return None

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
def writeVarsTxt(fname, varname, val, missing_value=None, delimiter=' ', usecols=None, fmt="%.10g"):
    """
    Write variables in a txt file in the following format.

    --- file (ascii) ---
    varname[0] varname[1] ... varname[nv-1]
    v[0, 0]    v[0, 1]    ... v[0, nv-1]
    v[1, 0]    v[1, 1]    ... v[1, nv-1]
    ...
    v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param fname:           (string or file handle) name of the file
    :param varname:         (list of string) variable names
    :param val:             (2d array) values of the variables, 2d array with
                                len(varname) columns, with
                                val[:,i] the values of variable of index i
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
    :param usecols:         (tuple or int or None) columns index (first column
                                is index 0) to be written, if None (default) all
                                columns are written
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'

    :return None:
    """

    funcname = 'writeVarsTxt'

    if not isinstance(varname, list):
        print(f'ERROR ({funcname}): varname invalid, should be a list')
        return None

    if val.ndim != 2 or val.shape[1] != len(varname):
        print(f'ERROR ({funcname}): val is incompatible with varname')
        return None

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
        get_sorting=False):
    """
    Reads grid geometry information, and sorting mode of filling (if asked for),
    from the header in a file.

    The grid geometry , i.e.
        (nx, ny, nz), grid size, number of cells along each direction,
        (sx, sy, sz), grid cell size along each direction,
        (ox, oy, oz), grid origin, coordinates of the bottom-lower-left corner
    is retrieved from the header (lines starting with the header_str identifier
    in the beginning of the file). Default values are used if not specified.

    The sorting mode (for filling the grid with the variables) is also retrieved
    if demanded. If not specified, the default mode is sorting='+X+Y+Z', which
    means that the grid is filled with
        x index increases, then y index increases, then z index increases
    The string sorting should have 6 characters (see exception below):
        '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'
    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with
        sorting[1] index decreases if sorting[0]='-', increases if sorting[0]='+'
    then
        sorting[3] index decreases if sorting[2]='-', increases if sorting[2]='+'
    then
        sorting[5] index decreases if sorting[4]='-', increases if sorting[4]='+'
    As an exception, if nz=1, the string sorting can have 4 characters:
        '[+|-][X|Y][+|-][X|Y]'
    it is then interpreted as above by appending '+Z'.
    Note: sorting string is case insensitive.
    Note: the validity of the string sorting is not checked in this function.

    Example of file:

    --- file (ascii) ---
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
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    Only the lines starting with the string header_str ('#' by default) in the
    beginning of the file are read, but at maximum max_lines lines (if None, not
    limited).

    :param filename:        (string) name of the file
    :param nx, ny, nz:      (int) default number of grid cells along each axis
    :param sx, sy, sz:      (float) default cell size along each axis
    :param ox, oy, oz:      (float) default origin of the grid
                                (bottom-lower-left corner)
    :param sorting:         (string) default, describes the way to fill the
                                grid
    :param header_str:      (string or None) only lines starting with that string
                                in the beginning of the file are treated (no
                                restriction if None)
    :param max_lines:       (int or None) maximum of lines read (unlimited if
                                None);
                                note: if header_str=None and max_lines=None,
                                the whole file will be read
    :param key_nx:          (list of string) possible key words
                                            (case insensitive) for entry nx
    :param key_ny:          (list of string) possible key words
                                            (case insensitive) for entry ny
    :param key_nz:          (list of string) possible key words
                                            (case insensitive) for entry nz
    :param key_sx:          (list of string) possible key words
                                            (case insensitive) for entry sx
    :param key_sy:          (list of string) possible key words
                                            (case insensitive) for entry sy
    :param key_sz:          (list of string) possible key words
                                            (case insensitive) for entry sz
    :param key_ox:          (list of string) possible key words
                                            (case insensitive) for entry ox
    :param key_oy:          (list of string) possible key words
                                            (case insensitive) for entry oy
    :param key_oz:          (list of string) possible key words
                                            (case insensitive) for entry oz
    :param key_sorting:     (list of string) possible key words
                                            (case insensitive) for entry sorting
    :param get_sorting:     (bool) indicates if sorting mode is retrieved (True)
                                or not (False)

    :return ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz)[, sorting]):
        grid geometry, where
            (nx, ny, nz)    (3-tuple of ints) number of cells along each axis,
            (sx, sy, sz)    (3-tuple of floats) cell size along each axis
            (ox, oy, oz)    (3-tuple of floats) coordinates of the origin
                                (bottom-lower-left corner)
        and (if get_sorting=True):
            sorting         (string) string of length 6 describing the sorting
                                mode of filling
    """

    fname = 'readGridInfoFromHeaderTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

    # Check header_str identifier
    if header_str is not None:
        if header_str == '':
            print(f'ERROR ({fname}): header_str identifier cannot be an empty string, use header_str=None instead')
            return None
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
                        print(f'ERROR ({fname}): more than one entry for "nx"')
                        return None
                    try:
                        nx = int(line_s[k+1])
                        k = k+2
                        nx_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "nx"')
                        return None

                elif entry in key_ny: # entry for ny ?
                    if ny_flag:
                        print(f'ERROR ({fname}): more than one entry for "ny"')
                        return None
                    try:
                        ny = int(line_s[k+1])
                        k = k+2
                        ny_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "ny"')
                        return None

                elif entry in key_nz: # entry for nz ?
                    if nz_flag:
                        print(f'ERROR ({fname}): more than one entry for "nz"')
                        return None
                    try:
                        nz = int(line_s[k+1])
                        k = k+2
                        nz_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "nz"')
                        return None

                elif entry in key_sx: # entry for sx ?
                    if sx_flag:
                        print(f'ERROR ({fname}): more than one entry for "sx"')
                        return None
                    try:
                        sx = float(line_s[k+1])
                        k = k+2
                        sx_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "sx"')
                        return None

                elif entry in key_sy: # entry for sy ?
                    if sy_flag:
                        print(f'ERROR ({fname}): more than one entry for "sy"')
                        return None
                    try:
                        sy = float(line_s[k+1])
                        k = k+2
                        sy_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "sy"')
                        return None

                elif entry in key_sz: # entry for sz ?
                    if sz_flag:
                        print(f'ERROR ({fname}): more than one entry for "sz"')
                        return None
                    try:
                        sz = float(line_s[k+1])
                        k = k+2
                        sz_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "sz"')
                        return None

                elif entry in key_ox: # entry for ox ?
                    if ox_flag:
                        print(f'ERROR ({fname}): more than one entry for "ox"')
                        return None
                    try:
                        ox = float(line_s[k+1])
                        k = k+2
                        ox_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "ox"')
                        return None

                elif entry in key_oy: # entry for oy ?
                    if oy_flag:
                        print(f'ERROR ({fname}): more than one entry for "oy"')
                        return None
                    try:
                        oy = float(line_s[k+1])
                        k = k+2
                        oy_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "oy"')
                        return None

                elif entry in key_oz: # entry for oz ?
                    if oz_flag:
                        print(f'ERROR ({fname}): more than one entry for "oz"')
                        return None
                    try:
                        oz = float(line_s[k+1])
                        k = k+2
                        oz_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "oz"')
                        return None

                elif entry in key_sorting and get_sorting: # entry for sorting (and get_sorting)?
                    if sorting_flag:
                        print(f'ERROR ({fname}): more than one entry for "sorting"')
                        return None
                    try:
                        sorting = line_s[k+1]
                        k = k+2
                        sorting_flag = True
                    except:
                        print(f'ERROR ({fname}): reading entry for "sorting"')
                        return None

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
        usecols=None):
    """
    Reads an image from a txt file, including grid geometry information, and
    sorting mode of filling.

    The grid geometry , i.e.
        (nx, ny, nz), grid size, number of cells along each direction,
        (sx, sy, sz), grid cell size along each direction,
        (ox, oy, oz), grid origin, coordinates of the bottom-lower-left corner
    is retrieved from the header (lines starting with the comments identifier in
    the beginning of the file). Default values are used if not specified.

    The number n of values (see below) for each variable should be equal to
    nx*ny*nz. The grid is filled according to the specified sorting mode.
    If not specified, the default mode is sorting='+X+Y+Z', which means that the
    grid is filled with
        x index increases, then y index increases, then z index increases
    The string sorting must have 6 characters (see exception below):
        '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'
    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with
        sorting[1] index decreases if sorting[0]='-', increases if sorting[0]='+'
    then
        sorting[3] index decreases if sorting[2]='-', increases if sorting[2]='+'
    then
        sorting[5] index decreases if sorting[4]='-', increases if sorting[4]='+'
    As an exception, if nz=1, the string sorting can have 4 characters:
        '[+|-][X|Y][+|-][X|Y]'
    it is then interpreted as above by appending '+Z'.
    Note: sorting string is case insensitive.

    Geometry grid information and sorting mode of filling is retrieved from the
    header of the file, i.e. the commented lines in the beginning of the file
    (see also function readGridInfoFromHeaderTxt).

    Example of file:

    --- file (ascii) ---
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
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param filename:        (string) name of the file
    :param nx, ny, nz:      (int) default number of grid cells along each axis
    :param sx, sy, sz:      (float) default cell size along each axis
    :param ox, oy, oz:      (float) default origin of the grid
                                (bottom-lower-left corner)
    :param sorting:         (string) default, describes the way to fill the
                                grid
    :param missing_value:   (float or None) value that will be replaced by nan
                                grid (see above)
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
                                Note: "empty field" after splitting is ignored,
                                then if white space is used as delimiter
                                (default), one or multiple white spaces can be
                                used as the same delimiter
    :param comments:        (string) lines starting with that string composed
                                the header of the file from which the grid
                                geometry is read
    :param usecols:         (tuple or int or None) columns index (first column
                                is index 0) to be read, if None (default) all
                                columns are read

    :return im: (Img class) image containing the variables that have been read
    """

    fname = 'readImageTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

    # Check comments identifier
    if comments is None or comments == '':
        print(f'ERROR ({fname}): comments cannot be an empty string (nor None)')
        return None

    # Read grid geometry information and sorting mode from header
    try:
        ((nx, ny, nz), (sx, sy, sz), (ox, oy, oz), sorting) = \
            readGridInfoFromHeaderTxt(
                filename,
                nx=nx, ny=ny, nz=nz,
                sx=sx, sy=sy, sz=sz,
                ox=ox, oy=oy, oz=oz,
                sorting=sorting,
                header_str=comments,
                get_sorting=True)
    except:
        print(f'ERROR ({fname}): grid geometry information cannot be read')
        return None

    # Deal with sorting
    if len(sorting) == 4:
        sorting = sorting + '+Z'

    if len(sorting) != 6:
        print(f'ERROR ({fname}): invalid sorting (string)')
        return None

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
        print(f'ERROR ({fname}): invalid sorting (string)')
        return None

    flip = [1, 1, 1]
    for i in range(3):
        s = sorting[2*i]
        if s == '-':
            flip[i] = -1
        elif s != '+':
            print(f'ERROR ({fname}): invalid sorting (string)')
            return None

    # Read variale names and values from file
    try:
        varname, val = readVarsTxt(filename, missing_value=missing_value, delimiter=delimiter, comments=comments, usecols=usecols)
    except:
        print(f'ERROR ({fname}): variables names / values cannot be read')
        return None

    if val.shape[0] != nx*ny*nz:
        print(f'ERROR ({fname}): number of grid cells and number of values for each variable differs')
        return None

    # Reorganize val array according to sorting, final shape: (len(varname), nz, ny, nx)
    val = val.T.reshape(-1, *sha)[:, ::flip[2], ::flip[1], ::flip[0]].transpose(0, *tr)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, len(varname), val, varname, filename)

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
        fmt="%.10g"):
    """
    Writes an image in a txt file, including grid geometry information, and
    sorting mode of filling.

    The grid geometry information and the sorting mode of filling is written
    in the beginning of the file with lines starting with the string comments.

    If not specified, the default mode is sorting='+X+Y+Z', which means that the
    grid is filled (with values as they are written) with
        x index increases, then y index increases, then z index increases
    The string sorting must have 6 characters (see exception below):
        '[+|-][X|Y|Z][+|-][X|Y|Z][+|-][X|Y|Z]'
    where 'X', 'Y', 'Z' appears exactly once, and has the following meaning: the
    grid is filled with
        sorting[1] index decreases if sorting[0]='-', increases if sorting[0]='+'
    then
        sorting[3] index decreases if sorting[2]='-', increases if sorting[2]='+'
    then
        sorting[5] index decreases if sorting[4]='-', increases if sorting[4]='+'
    As an exception, if nz=1, the string sorting can have 4 characters:
        '[+|-][X|Y][+|-][X|Y]'
    it is then interpreted as above by appending '+Z'.
    Note: sorting string is case insensitive.

    Example of file:

    --- file (ascii) ---
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
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param filename:        (string) name of the file
    :param im               (Img class) image to be written
    :param sorting:         (string) describes the sorting mode (see above)
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
    :param comments:        (string) that string is used in the beginning of each
                                line in the header (for writing grid geometry
                                information and sorting mode)
    :param endofline:       (string) string for end of line
    :param usevars:         (tuple or int or None) variable index(-es) to be
                                written
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'

    :return None:
    """

    fname = 'writeImageTxt'

    # Check comments identifier
    if comments is None or comments == '':
        print(f'ERROR ({fname}): comments cannot be an empty string (nor None)')
        return None

    if usevars is not None:
        if isinstance(usevars, int):
            if usevars < 0 or usevars >= im.nv:
                print(f'ERROR ({fname}): usevars invalid')
                return None
        else:
            if np.any([iv < 0 or iv >= im.nv for iv in usevars]):
                print(f'ERROR ({fname}): usevars invalid')
                return None

    # Deal with sorting
    if len(sorting) == 4:
        sorting = sorting + '+Z'

    if len(sorting) != 6:
        print(f'ERROR ({fname}): invalid sorting (string)')
        return None

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
        print(f'ERROR ({fname}): invalid sorting (string)')
        return None

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
            print(f'ERROR ({fname}): invalid sorting (string)')
            return None

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
        writeVarsTxt(ff, im.varname, val, missing_value=missing_value, delimiter=delimiter, usecols=usevars, fmt=fmt)

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
        x_def=0.0, y_def=0.0, z_def=0.0):
    """
    Reads a point set from a txt file.

    If the flag 'set_xyz_as_first_vars' is set to True, the x, y, z coordinates
    of the points are set as variables with index 0, 1, 2, in the output point
    set. The coordinates are identified by the names 'x', 'y', 'z' (case
    insensitive); if a coordinate is not present in the file, it is added as a
    variable in the output point set and set to the default value specified by
    'x_def', 'y_def', 'z_def' (for x, y, z) for all points. Moreover, the x, y, z
    coordinates are set as variables of index 0, 1, 2 respectively, by reordering
    the variables if needed.

    Example of file.

    --- file (ascii) ---
    # commented line ...
    # [...]
    varname[0] varname[1] ... varname[nv-1]
    v[0, 0]    v[0, 1]    ... v[0, nv-1]
    v[1, 0]    v[1, 1]    ... v[1, nv-1]
    ...
    v[n-1, 0]  v[n-1, 1]  ... v[n-1, nv-1]
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
                                Note: "empty field" after splitting is ignored,
                                then if white space is used as delimiter
                                (default), one or multiple white spaces can be
                                used as the same delimiter
    :param comments:        (string or None) lines starting with that string are
                                treated as comments
    :param usecols:         (tuple or int or None) columns index (first column
                                is index 0) to be read, if None (default) all
                                columns are read
    :param set_xyz_as_first_vars:
                            (bool) If True: the x, y, z coordinates are set as
                                variables of index 0, 1, 2 in the ouput point set
                                (adding them and reodering if needed, see above);
                                If False: the variables of the point set will be
                                exactly the columns of the file
    :param x_def, y_def, z_def:
                            (float) default values for x, y, z coordinates, used
                                if a coordinate is added as variable
                                (set_xyz_as_first_vars=True)

    :return ps: (PointSet class) point set
    """

    fname = 'readPointSetTxt'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

    # Read variale names and values from file
    try:
        varname, val = readVarsTxt(filename, missing_value=missing_value, delimiter=delimiter, comments=comments, usecols=usecols)
    except:
        print(f'ERROR ({fname}): variables names / values cannot be read')
        return None

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
            print(f"ERROR ({fname}): x-coordinates given more than once")
        else:
            ix = -1 # x-coordinates not given

        iy = np.where([vn in ('y', 'Y') for vn in varname])[0]
        if len(iy) == 1:
            iy = iy[0]
            ic.append(iy)
        elif len(iy) > 1:
            print(f"ERROR ({fname}): y-coordinates given more than once")
        else:
            iy = -1 # y-coordinates not given

        iz = np.where([vn in ('z', 'Z') for vn in varname])[0]
        if len(iz) == 1:
            iz = iz[0]
            ic.append(iz)
        elif len(iz) > 1:
            print(f"ERROR ({fname}): z-coordinates given more than once")
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
        fmt="%.10g"):
    """
    Writes a point set in a txt file.

    Example of file.

    --- file (ascii) ---
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
    --- file (ascii) ---

    where varname[j] (string) is a the name of the variable of index j, and
    v[i, j] (float) is the value of the variable of index j, for the entry of
    index i, i.e. one entry per line.

    :param filename:        (string) name of the file
    :param ps:              (PointSet class) point set to be written
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param delimiter:       (string) delimiter used to separate names and values
                                in each line
    :param comments:        (string) that string is used in the beginning of each
                                line in the header (for writing some info)
    :param endofline:       (string) string for end of line
    :param usevars:         (tuple or int or None) variable index(-es) to be
                                written
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'

    :return None:
    """

    fname = 'writePointSetTxt'

    # Check comments identifier
    if comments is None or comments == '':
        print(f'ERROR ({fname}): comments cannot be an empty string (nor None)')
        return None

    if usevars is not None:
        if isinstance(usevars, int):
            if usevars < 0 or usevars >= ps.nv:
                print(f'ERROR ({fname}): usevars invalid')
                return None
        else:
            if np.any([iv < 0 or iv >= ps.nv for iv in usevars]):
                print(f'ERROR ({fname}): usevars invalid')
                return None

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
        writeVarsTxt(ff, ps.varname, ps.val.T, missing_value=missing_value, delimiter=delimiter, usecols=usevars, fmt=fmt)

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImage2Drgb(filename, categ=False, nancol=None, keep_channels=True, rgb_weight=(0.299, 0.587, 0.114), flip_vertical=True):
    """
    Reads an image from a file using matplotlib.pyplot.imread, and fill a
    corresponding Img class instance. The image must be in 2D, with a RGB  or
    RGBA code for every pixel, the file format can be png, ppm, jpeg, etc. (e.g.
    created by Gimp).
    Note that every channel (RGB) is renormalized in [0, 1] by dividing by 255
    if needed.
    Treatement of colors (RGB or RGBA):
        - nancol is a color (RGB or RGBA) that is considered as "missing value",
            i.e. nan in the output image (Img class),
        - keep_channels: if True, every channel is retrieved (3 channels if RGB
            or 4 channels if RGBA); otherwise (False), the channels RGB (alpha
            channel, if present, is ignored) are linearly combined using the
            weights 'rgb_weight', to get color codes defined as one value in
            [0, 1].
    Type of image:
        - continuous (categ=False): the output image (Img class) has one
            variable if keep_channels is False, and 3 or 4 variables (resp. for
            colors as RGB or RGBA codes in input image) if keep_channels is True
        - categorical (categ=True): the list of distinct colors in the input
            image is retrieved (list col) and indexed (from 0); the output image
            (Img class) has one variable defined as the index of the color (in
            the list col); the list col is also retrieved in output (every
            entry is a unique value (keep_channels=Fase) or a sequence of length
            3 or 4 (keep_channels=True); the output image can be drawn (plotted)
            directly by using:
                - geone.imgplot.drawImage2D(im, categ=True, categCol=col),
                    if keep_channels is True
                - geone.imgplot.drawImage2D(im, categ=True,
                                            categCol=[cmap(c) for c in col]),
                    where cmap is a color map function defined on the interval
                    [0, 1], if keep_channels is False

    :param filename:        (string) name of the file
    :param categ:           (bool) indicating the type of output image:
                                - if True: "categorical" output image with one
                                    variable interpreted as an index
                                - if False: "continuous" output image
    :param nancol:          (3-tuple or None): RGB color code (alpha channel,
                                if present, is ignored) (or string), color
                                interpreted as missing value (nan) in output
                                image
    :param keep_channels:   (bool) for RGB or RGBA images:
                                - if True: keep every channel
                                - if False: first three channels (RGB) are
                                    linearly combined using the weight
                                    'rgb_weight', to define one variable (alpha
                                    channel, if present, is ignored)
    :param rgb_weight:      (3-tuple) weights for R, G, B channels used to
                                combine channels (used if keep_channels=False);
                                notes:
                                - by default: from Pillow image convert mode “L”
                                - other weights could be e.g.
                                    (0.2125, 0.7154, 0.0721)
    :param flip_vertical:   (bool) if True, the image is flipped vertically
                                after reading (this is useful because the
                                "origin" of the input image is considered at the
                                top left, whereas it is at bottom left in the
                                output image)

    :return out:    depends on 'categ':
                        - if categ is False:
                            out = im
                        - if categ is True:
                            out = (im, col)
                        with:
                            im :    (Img class) output image (see "Type of
                                        image" above)
                            col:    (sequence) of colors, each component is a
                                        unique value (in [0,1]) or a 3-tuple
                                        (RGB code) or a 4-tuple (RGBA code); the
                                        output image has one variable which is
                                        the index of the color
    """

    fname = 'readImage2Drgb'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

    # Read image
    vv = plt.imread(filename)

    # Reshape image: one pixel per line
    ny, nx = vv.shape[0:2]
    vv = vv.reshape(nx*ny, -1)
    nv = vv.shape[1]

    # Check input image
    if nv != 3 and nv != 4:
        print(f'ERROR ({fname}): the input image must be in RGB or RGBA (3 or 4 channels for each pixel)')
        return None

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
    im = Img(nx, ny, 1, nv=nv, val=vv, varname=varname)

    if categ:
        out = (im, col)
    else:
        out = im

    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImage2Drgb(filename, im, col=None, cmap='gray', nancol=(1.0, 0.0, 0.0), flip_vertical=True):
    """
    Writes (saves) an image from an Img class instance in a file (using
    matplotlib.pyplot.imsave), in format png, ppm, jpeg, etc. The input image
    (Img class) should be in 2D with one variable, 3 variables (channels RGB) or
    4 variables (channels RGBA).
    Treatement of colors (RGB or RGBA):
        - if the input image (Img class) has one variable, then:
            - if a list col of RGB (or RGBA) colors is given: the image variable
                must represent (integer) index in [0, len(col)-1]), then the
                colors from the list are used for every pixel according to the
                index (variable value) at each pixel
            - if col is None (not given), the color are set from the variable by
                using colormap cmap (defined on [0,1]);
        - if the input image (Img class) has 3 or 4 variables, then they are
            considered as RGB or RGBA color codes
        - nancol is the color (RGB or RGBA) used for missing value in input
            image (Img class)

    :param filename:        (string) name of the file
    :param im:              (Img class) image to be saved in file (input image)
    :param col:             (list or None) list of colors RGB (3-tuple) or
                                RGBA code (4-tuple), for each category of the
                                image: only for image with one variable with
                                integer values in [0, len(col)-1]
    :param cmap:            colormap (can be a string: in this case the color
                                map matplotlib.pyplot.get_cmap(cmap) is used),
                                only for image with one variable when col is
                                None
    :param nancol:          (3-tuple or 4-tuple) RGB or RGBA color code (or
                                string) used for missing value (nan) in input
                                image
    :param flip_vertical:   (bool) if True, the image is flipped vertically
                                before writing (this is useful because the
                                "origin" of the input image is considered at the
                                bottom left, whereas it is at top left in file
                                png, etc.)

    :return None:
    """

    fname = 'writeImage2Drgb'

    # Check image parameters
    if im.nz != 1:
        print(f"ERROR ({fname}): 'im.nz' must be 1")
        return None

    if im.nv not in [1, 3, 4]:
        print(f"ERROR ({fname}): 'im.nv' must be 1, 3, or 4")
        return None

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
                print(f'ERROR ({fname}): col must be a sequence of RGB or RBGA color (each entry is a sequence of length 3 or 4)')
                return None

            if not np.all(np.array([len(c) for c in col]) == nchan):
                print(f'ERROR ({fname}): same format is required for every color in col')
                return None

            # "format" nancol
            if nchan == 3:
                nancolf = mcolors.to_rgb(nancol)
            elif nchan == 4:
                nancolf = mcolors.to_rgba(nancol)
            else:
                print(f'ERROR ({fname}): invalid format for the colors (RGB or RGBA required)')
                return None

            # Check value in vv
            if np.any((vv < 0, vv >= len(col))):
                print(f'ERROR ({fname}): variable value in image cannot be treated as index in col')
                return None

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
                    print(f'ERROR ({fname}): invalid cmap string! (grayscale is used)')
                    cmap = plt.get_cmap("gray")

            if np.any((vv < 0, vv > 1)):
                print(f'WARNING ({fname}): variable values in image are not in interval [0,1], they are rescaled')
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
def readImageGslib(filename, missing_value=None):
    """
    Reads an image from a file (gslib format).

    --- file (ascii) ---
    Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
    nvar
    name_of_variable_1
    ...
    name_of_variable_nvar
    V1(0)    ... Vnvar(0)
    ...
    V1(n-1) ... Vnvar(n-1)
    --- file (ascii) ---

    where Vi(j) denotes the value of index j for the i-th variable.

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return im: (Img class) image containing the variables that have been read
    """

    fname = 'readImageGslib'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

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
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr.T, varname, filename)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageGslib(im, filename, missing_value=None, fmt="%.10g"):
    """
    Writes an image in a file (gslib format).

    --- file (ascii) ---
    Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
    nvar
    name_of_variable_1
    ...
    name_of_variable_nvar
    V1(0)    ... Vnvar(0)
    ...
    V1(n-1) ... Vnvar(n-1)
    --- file (ascii) ---

    where Vi(j) denotes the value of index j for the i-th variable.

    :param im:              (Img class) image to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'

    :return None:
    """

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
def readImageVtk(filename, missing_value=None):
    """
    Reads an image from a file (vtk format).

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return im: (Img class) image containing the variables that have been read
    """

    fname = 'readImageVtk'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

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
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr.T, varname, filename)

    return im
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageVtk(im, filename, missing_value=None, fmt="%.10g",
                  data_type='float', version=3.4, name=None):
    """
    Writes an image in a file (vtk format).

    :param im:              (Img class) image to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
    :param data_type:       (string) data type (can be 'float', 'int', ...)
    :param version:         (float) version number (for data file)
    :param name:            (string or None) name to be written at line 2
                                if None, im.name is used

    :return None:
    """

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

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(im.val, np.isnan(im.val), missing_value)

    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        ff.write(shead.encode())
        # Write variable values
        np.savetxt(ff, im.val.reshape(im.nv, -1).T, delimiter=' ', fmt=fmt)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readPointSetGslib(filename, missing_value=None):
    """
    Reads a point set from a file (gslib format):

    --- file (ascii) ---
    npoint
    nvar+3
    name_for_x_coordinate
    name_for_y_coordinate
    name_for_z_coordinate
    name_of_variable_1
    ...
    name_of_variable_nvar
    x(1)      y(1)      z(1)      Z1(1)      ... Znvar(1)
    ...
    x(npoint) y(npoint) z(npoint) Z1(npoint) ... Znvar(npoint)
    --- file (ascii) ---

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return ps: (PointSet class) point set
    """

    fname = 'readPointSetGslib'

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f'ERROR ({fname}): invalid filename ({filename})')
        return None

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
    Writes a point set in a file (gslib format).

    :param ps:              (PointSet class) point set to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'

    :return None:
    """

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
    print("   run the module 'geone.imgplot'...")


# === OLD BELOW ===

# # ----------------------------------------------------------------------------
# def pointToGridIndex(x, y, z, sx=1.0, sy=1.0, sz=1.0, ox=0.0, oy=0.0, oz=0.0):
#     """
#     Convert real point coordinates to index grid:
#
#     :param x, y, z:     (float) coordinates of a point
#     :param sx, sy, sz:  (float) cell size along each axis
#     :param ox, oy, oz:  (float) origin of the grid (bottom-lower-left corner)
#
#     :return: ix, iy, iz:
#                         (3-tuple) grid node index
#                             in x-, y-, z-axis direction respectively
#                             Warning: no check if the node is within the grid
#     """
#     jx = (x-ox)/sx
#     jy = (y-oy)/sy
#     jz = (z-oz)/sz
#
#     ix = int(jx)
#     iy = int(jy)
#     iz = int(jz)
#
#     # round to lower index if between two grid node
#     if ix == jx and ix > 0:
#         ix = ix - 1
#     if iy == jy and iy > 0:
#         iy = iy - 1
#     if iz == jz and iz > 0:
#         iz = iz - 1
#
#     return ix, iy, iz
# # ----------------------------------------------------------------------------
