#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'img.py'
author:         Julien Straubhaar
date:           jan-2018

Definition of classes for images and point sets (gslib), and relative
functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
class Img(object):
    """
    Defines an image as a 3D grid with variable(s) / attribute(s):
        nx, ny, nz: (int) number of grid cells in each direction
        sx, sy, sz: (float) cell size in each direction
        ox, oy, oz: (float) origin of the grid (bottom-lower-left corner)
        nv:         (int) number of variable(s) / attribute(s)
        val:        ((nv,nz,ny,nx) array) attribute(s) / variable(s) values
        varname:    (list of string (or string)) variable names
        name:       (string) name of the image
    """

    def __init__(self,
                 nx=0,   ny=0,   nz=0,
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
            print ('ERROR: val does not have an acceptable size')
            return

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
                print ('ERROR: varname has not an acceptable size')
                return

        self.name = name

    # ------------------------------------------------------------------------
    def __str__(self):
        """Returns name of the image: string representation of Image object"""
        return self.name
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
            return

        if varname is None:
            varname = "V{:d}".format(ii)
        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_dimension(self, nx, ny, nz, newval=np.nan):
        """
        Sets dimensions and update shape of values array (by possible
        truncation or extension):

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
            return

        if iy0 >= iy1:
            print("Nothing is done! (invalid indices along y)")
            return

        if iz0 >= iz1:
            print("Nothing is done! (invalid indices along z)")
            return

        if iv0 >= iv1:
            print("Nothing is done! (invalid indices along v)")
            return

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
        Inserts a variable at a given index:

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain nx*ny*nz values
        :param varname: (string or None) name of the new variable
        :param ind:     (int) index where the new variable is inserted
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            print("Nothing is done! (invalid index)")
            return

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size != self.nxyz():
            print ('ERROR: val does not have an acceptable size')
            return

        # Extend val
        self.val = np.concatenate((self.val[0:ii,...],
                                   valarr.reshape(1, self.nz, self.ny, self.nx),
                                   self.val[ii:,...]),
                                  0)
        # Extend varname list
        if varname is None:
            varname = "V{:d}".format(self.nv)
        self.varname.insert(ii, varname)

        # Update nv
        self.nv = self.nv + 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def append_var(self, val=np.nan, varname=None):
        """
        Appends one variable:

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain nx*ny*nz values
        :param varname: (string or None) name of the new variable
        """

        self.insert_var(val=val, varname=varname, ind=self.nv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=-1):
        """
        Removes one variable (of given index).
        """
        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        iv =[i for i in range(self.nv)]
        del iv[ii]
        self.val = self.val[iv,...]

        # Update varname list
        del self.varname[ii]

        # Update nv
        self.nv = self.nv - 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """
        Removes all variables.
        """

        # Update val array
        self.val = np.zeros((0, self.nz, self.ny, self.nx))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, val=np.nan, varname=None, ind=-1):
        """
        Sets one variable (of given index):

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain nx*ny*nz values
        :param varname: (string or None) name of the variable
        :param ind:     (int) index where the variable to be set
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.nxyz())
        elif valarr.size != self.nxyz():
            print ('ERROR: val does not have an acceptable size')
            return

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.nz, self.ny, self.nx)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, indlist):
        """
        Extracts variable(s) (of given index-es):

        :param indlist: (int or list of ints) index or list of index-es of the
                            variable(s) to be extracted (kept)
        """

        indlist = list(np.asarray(indlist).flatten())
        indlist = [self.nv + i if i < 0 else i for i in indlist]

        if sum([i >= self.nv or i < 0 for i in indlist]) > 0:
            print("Nothing is done! (invalid index list)")
            return

        # Update val array
        self.val = self.val[indlist,...]

        # Update varname list
        self.varname = [self.varname[i] for i in indlist]

        # Update nv
        self.nv = len(indlist)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique_one_var(self,ind=0):
        """
        Gets unique values of one variable (of given index):

        :param ind: (int) index of the variable

        :return:    (1-dimensional array) unique values of the variable
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        uval = [val for val in np.unique(self.val[ind,...]).reshape(-1)
                if ~np.isnan(val)]

        return (uval)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True):
        """
        Gets proportions (density or count) of unique values of one
        variable (of given index):

        :param ind:     (int) index of the variable
        :param density: (bool) computes densities if True and counts otherwise

        :return:    (list (of length 2) of 1-dimensional array) out:
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
            return

        uv, cv = np.unique(self.val[ind,...],return_counts=True)

        cv = cv[~np.isnan(uv)]
        uv = uv[~np.isnan(uv)]

        if density:
            cv = cv / np.sum(cv)

        return ([uv, cv])
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique(self):
        """
        Gets unique values among all variables:

        :return:    (1-dimensional array) unique values found in any variable
        """

        uval = [val for val in np.unique(self.val).reshape(-1)
                if ~np.isnan(val)]

        return (uval)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop(self, density=True):
        """
        Gets proportions (density or count) of unique values for each variable:

        :param density: (bool) computes densities if True and counts otherwise

        :return:    (list (of length 2) of array) out:
                        out[0]: (1-dimensional array) unique values found in
                                any variable
                        out[1]: ((self.nv, len(out[0])) array) densities or
                                counts of the unique values:
                                out[i,j]: density or count of the j-th unique
                                value for the i-th variable
        """

        uv_all = self.get_unique()
        n = len(uv_all)
        cv_all = np.zeros(shape=(self.nv, n))

        for i in range(self.nv):
            uv, cv = self.get_prop_one_var(ind=i, density=density)
            for j in range(n):
                cv_all[i, uv_all==uv[j]] = cv[j]

        return ([uv_all, cv_all])
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
        Permutes x and y directions.
        """
        newval = np.zeros((self.nv, self.nz, self.nx, self.ny))
        for i in range(self.nv):
            for j in range(self.nz):
                newval[i, j, :, :] = self.val[i, j, :, :].T

        self.val = newval
        ntmp, stmp, otmp = self.nx, self.sx, self.ox
        self.nx, self.sx, self.ox = self.ny, self.sy, self.oy
        self.ny, self.sy, self.oy = ntmp, stmp, otmp
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permxz(self):
        """
        Permutes x and z directions.
        """
        newval = np.zeros((self.nv, self.nx, self.ny, self.nz))
        for i in range(self.nv):
            for j in range(self.ny):
                newval[i, :, j, :] = self.val[i, :, j, :].T

        self.val = newval
        ntmp, stmp, otmp = self.nx, self.sx, self.ox
        self.nx, self.sx, self.ox = self.nz, self.sz, self.oz
        self.nz, self.sz, self.oz = ntmp, stmp, otmp
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def permyz(self):
        """
        Permutes y and z directions.
        """
        newval = np.zeros((self.nv, self.ny, self.nz, self.nx))
        for i in range(self.nv):
            for j in range(self.nx):
                newval[i, :, :, j] = self.val[i, :, :, j].T

        self.val = newval
        ntmp, stmp, otmp = self.ny, self.sy, self.oy
        self.ny, self.sy, self.oy = self.nz, self.sz, self.oz
        self.nz, self.sz, self.oz = ntmp, stmp, otmp
    # ------------------------------------------------------------------------

    def nxyzv(self):
        return (self.nx * self.ny * self.nz * self.nv)

    def nxyz(self):
        return (self.nx * self.ny * self.nz)

    def nxy(self):
        return (self.nx * self.ny)

    def nxz(self):
        return (self.nx * self.nz)

    def nyz(self):
        return (self.ny * self.nz)

    def xmin(self):
        return (self.ox)

    def ymin(self):
        return (self.oy)

    def zmin(self):
        return (self.oz)

    def xmax(self):
        return (self.ox + self.nx * self.sx)

    def ymax(self):
        return (self.oy + self.ny * self.sy)

    def zmax(self):
        return (self.oz + self.nz * self.sz)

    def x(self):
        """
        Returns 1-dimensional array of x coordinates.
        """
        return (self.ox + 0.5 * self.sx + self.sx * np.arange(self.nx))

    def y(self):
        """
        Returns 1-dimensional array of y coordinates.
        """
        return (self.oy + 0.5 * self.sy + self.sy * np.arange(self.ny))

    def z(self):
        """
        Returns 1-dimensional array of z coordinates.
        """
        return (self.oz + 0.5 * self.sz + self.sz * np.arange(self.nz))

    def vmin(self):
        return (np.nanmin(self.val.reshape(self.nv,self.nxyz()),axis=1))

    def vmax(self):
        return (np.nanmax(self.val.reshape(self.nv,self.nxyz()),axis=1))
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

        self.npt = int(npt)
        self.nv = int(nv)

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(npt*nv)
        elif valarr.size != npt*nv:
            print ('ERROR: val does not have an acceptable size')
            return

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
                print ('ERROR: varname has not an acceptable size')
                return

            self.varname = list(np.asarray(varname).reshape(-1))

        self.name = name

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
            return

        if varname is None:
            varname = "V{:d}".format(ii)
        self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, val=np.nan, varname=None, ind=0):
        """
        Inserts a variable at a given index:

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain npt values
        :param varname: (string or None) name of the new variable
        :param ind:     (int) index where the variable to be set
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            print("Nothing is done! (invalid index)")
            return

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            print ('ERROR: val does not have an acceptable size')
            return

        # Extend val
        self.val = np.concatenate((self.val[0:ii,...],
                                   valarr.reshape(1, self.npt),
                                   self.val[ii:,...]),
                                  0)
        # Extend varname list
        if varname is None:
            varname = "V{:d}".format(self.nv)
        self.varname.insert(ii, varname)

        # Update nv
        self.nv = self.nv + 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def append_var(self, val=np.nan, varname=None):
        """
        Appends one variable:

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain npt values
        :param varname: (string or None) name of the new variable
        """

        self.insert_var(val=val, varname=varname, ind=self.nv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=-1):
        """
        Removes one variable (of given index).
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        iv =[i for i in range(self.nv)]
        del iv[ii]
        self.val = self.val[iv,...]

        # Update varname list
        del self.varname[ii]

        # Update nv
        self.nv = self.nv - 1
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
        Sets one variable (of given index):

        :param val:     (int/float or tuple/list/ndarray) value(s) of the new
                            variable:
                            if type is int/float: constant variable
                            if tuple/list/ndarray: must contain npt values
        :param varname: (string or None) name of the new variable
        :param ind:     (int) index where the variable to be set
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        valarr = np.asarray(val, dtype=float) # numpy.ndarray (possibly 0-dimensional)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            print ('ERROR: val does not have an acceptable size')
            return

        # Set variable of index ii
        self.val[ii,...] = valarr.reshape(self.npt)

        # Set variable name of index ii
        if varname is not None:
            self.varname[ii] = varname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, indlist):
        """
        Extracts variable(s) (of given index-es):

        :param indlist: (int or list of ints) index or list of index-es of the
                            variable(s) to be extracted (kept)
        """

        indlist = list(np.asarray(indlist).flatten())
        indlist = [self.nv + i if i < 0 else i for i in indlist]

        if sum([i >= self.nv or i < 0 for i in indlist]) > 0:
            print("Nothing is done! (invalid index list)")
            return

        # Update val array
        self.val = self.val[indlist,...]

        # Update varname list
        self.varname = [self.varname[i] for i in indlist]

        # Update nv
        self.nv = len(indlist)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique(self,ind=0):
        """
        Gets unique values of one variable (of given index):

        :param ind: (int) index of the variable

        :return:    (1-dimensional array) unique values of the variable
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        return (np.unique(self.val[ind,...]))
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True):
        """
        Gets proportions (density or count) of unique values of one
        variable (of given index):

        :param ind:     (int) index of the variable
        :param density: (bool) computes densities if True and counts otherwise

        :return:    (list (of length 2) of 1-dimensional array) out:
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
            return

        uv, cv = list(np.unique(self.val[ind,...],return_counts=True))

        cv = cv[~np.isnan(uv)]
        uv = uv[~np.isnan(uv)]

        if density:
            cv = cv / np.sum(cv)

        return ([uv, cv])
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def to_dict(self):
        """
        Returns PointSet as a dictionary
        """
        return {name: values for name, values in zip(self.varname, self.val)}
    # ------------------------------------------------------------------------

    def x(self):
        return(self.val[0])

    def y(self):
        return(self.val[1])

    def z(self):
        return(self.val[2])

    def xmin(self):
        return (np.min(self.val[0]))

    def ymin(self):
        return (np.min(self.val[1]))

    def zmin(self):
        return (np.min(self.val[2]))

    def xmax(self):
        return (np.max(self.val[0]))

    def ymax(self):
        return (np.max(self.val[1]))

    def zmax(self):
        return (np.max(self.val[2]))
# ============================================================================

# ----------------------------------------------------------------------------
def copyImg(im, varIndList=None):
    """
    Copies an image (Img class), with all variables or a subset of variables:

    :param im:          (Img class) input image
    :param varIndList:  (sequence of int or None) index-es of the variables
                            to copy (default None: all variables), note that for
                            copying one variable, specify "varIndList=(iv,)"
    :return:            (Img class) a copy of the input image
                            (not a reference to)
    """

    if varIndList is not None:
        # Check if each index is valid
        if sum([iv in range(im.nv) for iv in varIndList]) != len(varIndList):
            print("ERROR: invalid index-es")
            return
    else:
        varIndList = range(im.nv)

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=len(varIndList),
                name=im.name)

    for i, iv in enumerate(varIndList):
        imOut.set_var(val=im.val[iv,...], varname=im.varname[iv], ind=i)

    return (imOut)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageGslib(filename, missing_value=None):
    """
    Reads an image from a file (gslib format):

    --- file (ascii) ---
    Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
    nvar
    name_of_variable_1
    ...
    name_of_variable_nvar
    Z1(0)    ... Znvar(0)
    ...
    Z1(Ng-1) ... Znvar(Ng-1)
    --- file (ascii) ---

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return:    (Img class) image
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("ERROR: invalid filename ({})".format(filename))
        return

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

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageVtk(filename, missing_value=None):
    """
    Reads an image from a file (vtk format):

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return:    (Img class) image
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("ERROR: invalid filename ({})".format(filename))
        return

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

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImagePgm(filename, missing_value=None, varname=['pgmValue']):
    """
    Reads an image from a file (pgm format):

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return:    (Img class) image
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("ERROR: invalid filename ({})".format(filename))
        return

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read 1st line
        line = ff.readline()
        if line[:2] != 'P2':
            print("ERROR: invalid format (first line)")
            return

        # Read 2nd line
        line = ff.readline()
        while line[0] == '#':
            # Read next line
            line = ff.readline()

        # Set dimension
        nx, ny = [int(x) for x in line.split()]

        # Read next line
        line = ff.readline()
        if line[:3] != '255':
            print("ERROR: invalid format (number of colors / max val)")
            return

        # Read the rest of the file
        vv = [x.split() for x in ff.readlines()]

    # Set grid
    nz = 1 # nx, ny already set
    sx, sy, sz = [1.0, 1.0, 1.0]
    ox, oy, oz = [0.0, 0.0, 0.0]

    # Set variable
    nv = 1
    varname # given in arguments

    # Set variable array
    valarr = np.array([int(x) for line in vv for x in line], dtype=float).reshape(-1, nv)

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr, varname, filename)

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImagePpm(filename, missing_value=None, varname=['ppmR', 'ppmG', 'ppmB']):
    """
    Reads an image from a file (ppm format):

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return:    (Img class) image
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("ERROR: invalid filename ({})".format(filename))
        return

    # Open the file in read mode
    with open(filename,'r') as ff:
        # Read 1st line
        line = ff.readline()
        if line[:2] != 'P3':
            print("ERROR: invalid format (first line)")
            return

        # Read 2nd line
        line = ff.readline()
        while line[0] == '#':
            # Read next line
            line = ff.readline()

        # Set dimension
        nx, ny = [int(x) for x in line.split()]

        # Read next line
        line = ff.readline()
        if line[:3] != '255':
            print("ERROR: invalid format (number of colors / max val)")
            return

        # Read the rest of the file
        vv = [x.split() for x in ff.readlines()]

    # Set grid
    nz = 1 # nx, ny already set
    sx, sy, sz = [1.0, 1.0, 1.0]
    ox, oy, oz = [0.0, 0.0, 0.0]

    # Set variable
    nv = 3
    varname # given in arguments

    # Set variable array
    valarr = np.array([int(x) for line in vv for x in line], dtype=float).reshape(-1, nv)

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set image
    im = Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv, valarr.T, varname, filename)

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imCategFromPgm(filename, flip_vertical=True, cmap='binary'):
    """
    Reads an image from a file (pgm format (ASCII), e.g. created by Gimp):

    :param filename:        (string) name of the file
    :param flip_vertical:   (bool) if True: flip the image vertically after reading the image

    :return:    (tuple) (im, code, col)
                    im: (Img class) image with categories 0, 1, ..., n-1 as values
                    col  : list of colors (rgba tuple, for each category) (length n)
                    pgm  : list of initial pgm values (length n)
    """

    # Read image
    im = img.readImagePgm(filename)

    if flip_vertical:
        # Flip image vertically
        im.flipy()

    # Set cmap function
    if isinstance(cmap, str):
        cmap_func = plt.get_cmap(cmap)
    else:
        cmap_func = cmap

    # Get colors and set color codes
    v = im.val.reshape(-1)
    pgm, code = np.unique(v, return_inverse=True)
    col = [cmap_func(c/255.) for c in pgm]

    # Set image
    im = Img(im.nx, im.ny, im.nz, im.sx, im.sy, im.sz, im.ox, im.oy, im.oz, nv=1, val=code, varname='code')

    return (im, col, pgm)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imCategFromPpm(filename, flip_vertical=True):
    """
    Reads an image from a file (ppm format (ASCII), e.g. created by Gimp):

    :param filename:        (string) name of the file
    :param flip_vertical:   (bool) if True: flip the image vertically after reading the image

    :return:    (tuple) (im, col)
                    im  : (Img class) image with categories 0, 1, ..., n-1 as values
                    col : list of colors (rgb[a] tuple (values in [0,1]), for each category) (length n)
                          Note: considering the image categorical, it can be drawn (plotted) directly
                            by using: geone.imgplot.drawImage2D(im, categ=True, categCol=col)
    """

    # Read image
    im = img.readImagePpm(filename)

    if flip_vertical:
        # Flip image vertically
        im.flipy()

    # Get colors and set color codes
    vv = im.val.reshape(im.nv, -1).T # array where each line is a color code (rgb[a])
    col, code = np.unique(vv, axis=0, return_inverse=True)
    col = list(col/255.)

    # # Get colors and set color codes
    # v = np.array((1, 256, 256**2)).dot(im.val.reshape(3,-1))
    # x, code = np.unique(v, return_inverse=True)
    # x,     ired   = np.divmod(x, 256)
    # iblue, igreen = np.divmod(x, 256)
    # rgb = np.array((ired, igreen, iblue)).T
    # col = [[c/255. for c in irgb] for irgb in rgb]

    # Set image
    im = Img(im.nx, im.ny, im.nz, im.sx, im.sy, im.sz, im.ox, im.oy, im.oz, nv=1, val=code, varname='code')

    return (im, col)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def readImageCateg(filename, flip_vertical=True):
    """
    Reads an image from a file (ppm (raw), pgm (raw), png format, e.g. created by Gimp) (using plt.imread):

    :param filename:        (string) name of the file
    :param flip_vertical:   (bool) if True: flip the image vertically after reading the image

    :return:    (tuple) (im, col)
                    im  : (Img class) image with categories 0, 1, ..., n-1 as values
                    col : list of colors (rgb[a] tuple (values in [0,1]), for each category) (length n)
                          Note: considering the image categorical, it can be drawn (plotted) directly
                            by using: geone.imgplot.drawImage2D(im, categ=True, categCol=col)
    """

    # Read image
    vv = plt.imread(filename)
    ny, nx = vv.shape[0:2]

    # Get colors and set color codes
    if len(vv.shape) == 2: # pgm image
        col, code = np.unique(vv, return_inverse=True)
        col = [1/255.*np.array([i, i, i]) for i in col] # gray scale

    else: # vv.shape of length 3 (ppm, png image)
        vv = vv.reshape(-1, vv.shape[-1]) # array where each line is a color code (rgb[a])
        col, code = np.unique(vv, axis=0, return_inverse=True)
        if col.dtype == 'uint8':
            col = col/255.
        col = list(col)

    if flip_vertical:
        code = code.reshape(ny, nx)
        code = code[::-1,:] # vertical flip

    # Set image
    im = Img(nx, ny, 1, nv=1, val=code, varname='code')

    return (im, col)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImageGslib(im, filename, missing_value=None, fmt="%.10g"):
    """
    Writes an image in a file (gslib format):

    :param im:              (Img class) image to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
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
def writeImageVtk(im, filename, missing_value=None, fmt="%.10g",
                  data_type='float', version=3.4, name=None):
    """
    Writes an image in a file (vtk format):

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
def writeImagePgm(im, filename, missing_value=None, fmt="%.10g"):
    """
    Writes an image in a file (pgm format):

    :param im:              (Img class) image to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
    """

    # Write 1st line in string shead
    shead = "P2\n# {0} {1} {2}   {3} {4} {5}   {6} {7} {8}\n{0} {1}\n255\n".format(
            im.nx, im.ny, im.nz, im.sx, im.sy, im.sz, im.ox, im.oy, im.oz)

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
def writeImagePpm(im, filename, missing_value=None, fmt="%.10g"):
    """
    Writes an image in a file (ppm format):

    :param im:              (Img class) image to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
    """

    # Write 1st line in string shead
    shead = "P3\n# {0} {1} {2}   {3} {4} {5}   {6} {7} {8}\n{0} {1}\n255\n".format(
            im.nx, im.ny, im.nz, im.sx, im.sy, im.sz, im.ox, im.oy, im.oz)

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
def isImageDimensionEqual (im1, im2):
    """
    Checks if grid dimensions of two images are equal.
    """

    return (im1.nx == im2.nx and im1.ny == im2.ny and im1.nz == im2.nz)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def gatherImages (imlist, varInd=None, remVarFromInput=False):
    """
    Gathers images:

    :param imlist:  (list) images to be gathered, they should have
                        the same grid dimensions
    :param varInd:  (int or None)
                        if None: all variables of each image from imlist
                            are put in the output image
                        else: only the variable of index varInd is put in
                            the output image
    :param remVarFromInput: (bool) if True, gathered variables are removed
                                from the source (input image)

    :return: (Img class) output image containing variables of images of imlist
    """

    for i in range(1,len(imlist)):
        if not isImageDimensionEqual(imlist[0], imlist[i]):
            print("ERROR: grid dimensions differ, nothing done!")
            return

    if varInd is not None:
        if varInd < 0:
            print("ERROR: invalid index (negative), nothing done!")
            return

        for i in range(len(imlist)):
            if varInd >= imlist[i].nv:
                print("ERROR: invalid index, nothing done!")
                return

    im = Img(nx=imlist[0].nx, ny=imlist[0].ny, nz=imlist[0].nz,
             sx=imlist[0].sx, sy=imlist[0].sy, sz=imlist[0].sz,
             ox=imlist[0].ox, oy=imlist[0].oy, oz=imlist[0].oz,
             nv=0, val=0.0)

    if varInd is not None:
        for i in range(len(imlist)):
            im.append_var(val=imlist[i].val[varInd,...])

            if remVarFromInput:
                imlist[i].remove_var(varInd)

    else:
        for i in range(len(imlist)):
            for j in range(imlist[i].nv):
                im.append_var(val=imlist[i].val[j,...])

            if remVarFromInput:
                imlist[i].remove_allvar()

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageContStat (im, op='mean', **kwargs):
    """
    Computes "pixel-wise" statistics over every variable of an image:

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
    :param kwargs:  additional key word arguments passed to np.<op>
                        function, typically: ddof=1 if op is 'std' or 'var'

    :return: (Img class) image with same grid as the input image and
                one variable being the pixel-wise statistics according to 'op'
                over every variable of the input image
    """

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
            print("ERROR: keyword argument 'q' required for op='quantile', nothing done!")
            return
        varname = [op + '_' + str(v) for v in kwargs['q']]
    else:
        print("ERROR: unkown operation {}, nothing done!".format(op))
        return

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
             sx=im.sx, sy=im.sy, sz=im.sz,
             ox=im.ox, oy=im.oy, oz=im.oz,
             nv=0, val=0.0)

    vv = func(im.val.reshape(im.nv,-1), axis=0, **kwargs)
    vv = vv.reshape(-1, im.nxyz())
    for v, name in zip(vv, varname):
        imOut.append_var(v, varname=name)

    return (imOut)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageCategProp (im, categ):
    """
    Computes "pixel-wise" proportions of given categories over every
    variable of an image:

    :param im:      (Img class) input image
    :param categ:   (sequence) list of value(s) for which the proportions
                        are computed

    :return: (Img class) image with same grid as the input image and as many
                variable(s) as given by 'categ', being the pixel-wise
                proportions of each category in 'categ', over every variable
                of the input image
    """

    categarr = np.array(categ,dtype=float).reshape(-1)

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=0, val=0.0)

    for code in categarr:
        x = im.val.reshape(im.nv,-1) == code
        x = np.asarray(x,dtype=float)
        np.putmask(x, np.isnan(im.val.reshape(im.nv,-1)), np.nan)
        imOut.append_var(np.mean(x, axis=0))

    return (imOut)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imageEntropy (im, varIndList=None):
    """
    Computes "pixel-wise" entropy for proprotions given as variables in an
    image:

    :param im:          (Img class) input image
    :param varIndList:  (sequence of int or None) index-es of the variables
                            to take into account (default None: all variables)
    :return:            (Img class) an image with one variable containing the
                            entropy for the variable given in input, at pixel i,
                            it is defined as:
                                Ent(i) = - sum_{v} p_v(i) * log_n(p(v(i)))
                            where v loops on each variable and n is the number
                            of variables. Note that the sum_{v} p(v(i)) should
                            be equal to 1
    """

    if varIndList is not None:
        # Check if each index is valid
        if sum([iv in range(im.nv) for iv in varIndList]) != len(varIndList):
            print("ERROR: invalid index-es")
            return
    else:
        varIndList = range(im.nv)

    if len(varIndList) < 2:
        print("ERROR: at least 2 indexes should be given")
        return

    imOut = Img(nx=im.nx, ny=im.ny, nz=im.nz,
                sx=im.sx, sy=im.sy, sz=im.sz,
                ox=im.ox, oy=im.oy, oz=im.oz,
                nv=1, val=np.nan,
                name=im.name)

    t = 1. / np.log(len(varIndList))

    for iz in range(im.nz):
        for iy in range(im.ny):
            for ix in range(im.nx):
                s = 0
                e = 0
                ok = True
                for iv in varIndList:
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

    return (imOut)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def copyPointSet(ps, varIndList=None):
    """
    Copies point set, with all variables or a subset of variables:

    :param ps:          (PointSet class) input point set
    :param varIndList:  (sequence of int or None) index-es of the variables
                            to copy (default None: all variables), note that for
                            copying one variable, specify "varIndList=(iv,)"
    :return:            (PointSet class) a copy of the input point set
                            (not a reference to)
    """

    if varIndList is not None:
        # Check if each index is valid
        if sum([iv in range(ps.nv) for iv in varIndList]) != len(varIndList):
            print("ERROR: invalid index-es")
            return
    else:
        varIndList = range(ps.nv)

    psOut = PointSet(npt=ps.npt, nv=len(varIndList), val=0.0, name=ps.name)

    for i, iv in enumerate(varIndList):
        psOut.set_var(val=ps.val[iv,...], varname=ps.varname[iv], ind=i)

    return (psOut)
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

    :return:                (PointSet class) point set
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("ERROR: invalid filename ({})".format(filename))
        return

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

    return (ps)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writePointSetGslib(ps, filename, missing_value=None, fmt="%.10g"):
    """
    Writes a point set in a file (gslib format):

    :param ps:              (PointSet class) point set to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
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

# ----------------------------------------------------------------------------
def imageToPointSet(im):
    """
    Returns a point set corresponding to the input image:

    :param im: (Img class) input image

    :return: (PointSet class) point set corresponding to the input image
    """

    # Initialize point set
    ps = PointSet(npt=im.nxyz(), nv=3+im.nv, val=0.0)

    # Set x-coordinate
    t = im.x()
    v = []
    for i in range(im.nyz()):
        v.append(t)

    ps.set_var(val=v, varname='X', ind=0)

    # Set y-coordinate
    t = np.repeat(im.y(), im.nx)
    v = []
    for i in range(im.nz):
        v.append(t)

    ps.set_var(val=v, varname='Y', ind=1)

    # Set z-coordinate
    v = np.repeat(im.z(), im.nxy())
    ps.set_var(val=v, varname='Z', ind=2)

    # Set next variable(s)
    for i in range(im.nv):
        ps.set_var(val=im.val[i,...], varname=im.varname[i], ind=3+i)

    return (ps)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pointSetToImage(ps, nx, ny, nz, sx=1.0, sy=1.0, sz=1.0, ox=0.0, oy=0.0, oz=0.0, job=0):
    """
    Returns an image corresponding to the input point set and grid:

    :param ps:  (PointSet class) input point set, with x, y, z-coordinates as
                    first three variable
    :param nx, ny, nz: (int) number of grid cells in each direction
    :param sx, sy, sz: (float) cell size in each direction
    :param ox, oy, oz: (float) origin of the grid (bottom-lower-left corner)
    :param job: (int)
                    if 0: an error occurs if one data is located outside of the
                        image grid, otherwise all data are integrated in the
                        image
                    if 1: data located outside of the image grid are ignored
                        (no error occurs), and all data located within the
                        image grid are integrated in the image

    :return: (Img class) image corresponding to the input point set and grid
    """

    if ps.nv < 3:
        print("ERROR: invalid number of variable (should be > 3)")
        return

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
    # iout = np.any([np.array(ix < 0), np.array(ix >= nx),
    #                np.array(iy < 0), np.array(iy >= ny),
    #                np.array(iz < 0), np.array(iz >= nz)],
    #               0)
    iout = np.any(np.array((ix < 0, ix >= nx,
                            iy < 0, iy >= ny,
                            iz < 0, iz >= nz)), 0)

    if not job and sum(iout) > 0:
        print ("ERROR: point out of the image grid!")
        return

    # Set values in the image
    for i in range(ps.npt): # ps.npt is equal to iout.size
        if not iout[i]:
            im.val[:,iz[i], iy[i], ix[i]] = ps.val[3:ps.nv,i]

    return (im)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pointToGridIndex(x, y, z, sx=1.0, sy=1.0, sz=1.0, ox=0.0, oy=0.0, oz=0.0):
    """
    Convert real point coordinates to index grid:

    :param x, y, z:     (float) coordinates of a point
    :param nx, ny, nz:  (int) number of grid cells in each direction
    :param sx, sy, sz:  (float) cell size in each direction
    :param ox, oy, oz:  (float) origin of the grid (bottom-lower-left corner)

    :return: ix, iy, iz:
                        (3-tuple) grid node index
                            in x-, y-, z-axis direction respectively
                            Warning: no check if the node is within the grid
    """

    jx = (x-ox)/sx
    jy = (y-oy)/sy
    jz = (z-oz)/sz

    ix = int(jx)
    iy = int(jy)
    iz = int(jz)

    # round to lower index if between two grid node
    if ix == jx and ix > 0:
        ix = ix -1
    if iy == jy and iy > 0:
        iy = iy -1
    if iz == jz and iz > 0:
        iz = iz -1

    return ix, iy, iz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz):
    """
    Convert a grid index (3 indices) into a single grid index:

    :param ix, iy, iz:  (int) grid index in x-, y-, z-axis direction
    :param nx, ny, nz:  (int) number of grid cells in each direction

    :return: i:         (int) single grid index
                            Note: ix, iy, iz can be ndarray of same shape, then
                            i in output is ndarray of that shape
    """
    return ix + nx * (iy + ny * iz)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def singleGridIndexToGridIndex(i, nx, ny, nz):
    """
    Convert a single into a grid index (3 indices):

    :param i:           (int) single grid index
    :param nx, ny, nz:  (int) number of grid cells in each direction

    :return: ix, iy, iz:
                        (3-tuple) grid index in x-, y-, z-axis direction
                            Note: i can be a ndarray, then
                            ix, iy, iz in output are ndarray (of same shape)
    """
    nxy = nx*ny
    iz = i//nxy
    j = i%nxy
    iy = j//nx
    ix = j%nx

    return ix, iy, iz
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sampleFromPointSet(point_set, size, seed=None, mask=None):
    """
    Sample random points from PointSet object
    and return a PointSet
    :param point_set: (PointSet) PointSet object to sample from
    :param size: (size) number of points to be sampled
    :param seed: (int) optional random seed
    :param mask: (PointSet) PointSet of the same size showing where to sample
                 points where mask == 0 will be not taken into account
    :return: PointSet:
                       A PointSet object
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
    Sample random points from Img object
    and return a PointSet
    :param image: (Img) Img object to sample from
    :param size:  (int) number of points to be sampled
    :param seed:  (int) optional random seed
    :param mask:  (Image) Image of the same size indicating where to sample
                  points where mask == 0 will be not taken into account
    :return: PointSet:
                       A PointSet object
    """
    # Create point set from image
    point_set = imageToPointSet(image)
    if mask is not None:
        mask = imageToPointSet(mask)

    return sampleFromPointSet(point_set, size, seed, mask)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def extractRandomPointFromImage (im, npt, seed=None):
    """
    Extracts random points from an image (at center of grid cells) and return
    the corresponding point set:

    :param im:  (Img class) input image
    :param npt: (int) number of points to extract
                    (if greater than the number of image grid cells,
                    npt is set to this latter)
    :seed:      (int) seed number for initializing the random number generator (if not None)

    :return:    (PointSet class) point set containing the extracting points
    """

    if npt <= 0:
        print("ERROR: number of points negative or zero (npt={}), nothing done!".format(npt))
        return

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

    return (ps)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.img' example:")
    print("   run the module 'geone.imgplot'...")
