#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'imgplot.py'
# author:         Julien Straubhaar
# date:           dec-2017
# -------------------------------------------------------------------------

"""
Module for custom plots of images (class :class:`geone.img.Img`) in 2D.
"""

from geone import img
from geone.img import Img
from geone import customcolors as ccol
from geone.customcolors import add_colorbar

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams as mpl_rcParams
import matplotlib.colors as mcolors

# ============================================================================
class ImgplotError(Exception):
    """
    Custom exception related to `imgplot` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def drawImage2D(
        im, ix=None, iy=None, iz=None, iv=None,
        plot_empty_grid=False,
        cmap=ccol.cmap_def,
        alpha=None,
        excludedVal=None,
        categ=False, categVal=None,
        categCol=None, categColCycle=False, categColbad=ccol.cbad_def,
        nCategMax=100,
        vmin=None, vmax=None,
        contourf=False,
        contour=False,
        contour_clabel=False,
        levels=None,
        contourf_kwargs=None,
        contour_kwargs={'linestyles':'solid', 'colors':'gray'},
        contour_clabel_kwargs={'inline':1},
        interpolation='none',
        aspect='equal',
        frame=True, xaxis=True, yaxis=True,
        title=None,
        xlabel=None, xticks=None, xticklabels=None, xticklabels_max_decimal=None,
        ylabel=None, yticks=None, yticklabels=None, yticklabels_max_decimal=None,
        clabel=None, cticks=None, cticklabels=None, cticklabels_max_decimal=None,
        colorbar_extend='neither',
        colorbar_aspect=20, colorbar_pad_fraction=1.0,
        showColorbar=True, removeColorbar=False, showColorbarOnly=0,
        verbose=1,
        logger=None,
        **kwargs):
    # animated : bool, default: False
    #     keyword argument passed to `matplotlib.pyplot.imshow` for animation...
    """
    Displays a 2D image (or a slice of a 3D image).

    Parameters
    ----------
    im : :class:`geone.img.Img`
        image

    ix : int, optional
        grid index along x axis of the yz-slice to be displayed;
        by default (`None`): no slice along x axis

    iy : int, optional
        grid index along y axis of the xz-slice to be displayed;
        by default (`None`): no slice along y axis

    iz : int, optional
        grid index along z axis of the xy-slice to be displayed;
        by default (`None`): no slice along z axis, but
        if `ix=None`, `iy=None`, `iz=None`, then `iz=0` is used

    iv : int, optional
        index of the variable to be displayed;
        by default (`None`): the variable of index 0 (`iv=0` is used) if
        the image has at leas one variable (`im.nv > 0`), otherwise, an "empty"
        grid is displayed (a "fake" variable with `numpy.nan` (missing value)
        over the entire grid is considered)

    plot_empty_grid : bool, default: False
        if `True`: an "empty" grid is displayed (a "fake" variable with `numpy.nan`
        (missing value) over the entire grid is considered), and `iv` is ignored

    cmap : colormap
        color map (can be a string, in this case the color map is obtained by
        `matplotlib.pyplot.get_cmap(cmap)`)

    alpha : float, optional
        value of the "alpha" channel (for transparency);
        by default (`None`): `alpha=1.0` is used (no transparency)

    excludedVal : sequence of values, or single value, optional
        values to be excluded from the plot;
        note not used if `categ=True` and `categVal` is not `None`

    categ : bool, default: False
        indicates if the variable of the image to diplay has to be treated as a
        categorical (discrete) variable (`True`), or continuous variable (`False`)

    categVal : sequence of values, or single value, optional
        used only if `categ=True`:
        explicit list of the category values to be displayed;
        by default (`None`): the list of all unique values are automatically
        retrieved

    categCol: sequence of colors, optional
        used only if `categ=True`:
        sequence of colors, (given as 3-tuple (RGB code), 4-tuple (RGBA code) or
        str), used for the category values that will be displayed:

        - if `categVal` is not `None`: `categCol` must have the same length as \
        `categVal`
        - if `categVal=None`:
            - first colors of `categCol` are used if its length is greater than \
            or equal to the number of displayed category values,
            - otherwise: the colors of `categCol` are used cyclically if \
            `categColCycle=True`, and the colors taken from the color map `cmap` \
            are used if `categColCycle=False`

    categColCycle : bool, default: False
        used only if `categ=True`:
        indicates if the colors of `categCol` can be used cyclically or not
        (when the number of displayed category values exceeds the length of
        `categCol`)

    categColbad : color
        used only if `categ=True`:
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str) used for bad
        categorical value

    nCategMax : int, default: 100
        used only if `categ=True`: maximal number of categories, if there is 
        more distinct values to be plotted, an error is raised
        
    vmin : float, optional
        used only if `categ=False`:
        minimal value to be displayed; by default: minimal value of the displayed
        variable is used for `vmin`

    vmax : float, optional
        used only if `categ=False`:
        maximal value to be displayed; by default: maximal value of the displayed
        variable is used for `vmax`

    contourf : bool, default: False
        indicates if `matplotlib.pyplot.contourf` is used, i.e. contour map with
        filled area between levels, instead of standard plot
        (`matplotlib.pyplot.imshow`)

    contour : bool, default: False
        indicates if contour levels are added to the plot (using
        `matplotlib.pyplot.contour`)

    contour_clabel : bool, default: False
        indicates if labels are added to contour (if `contour=True`)

    levels : array-like, or int, optional
        keyword argument 'levels' passed to `matplotlib.pyplot.contourf` (if
        `contourf=True`) and/or `matplotlib.pyplot.contour` (if `contour=True`)

    contourf_kwargs : dict, optional
        keyword arguments passed to `matplotlib.pyplot.contourf` (if
        `contourf=True`); note: the parameters `levels` (see above) is used as
        keyword argument, i.e. it prevails over the key 'levels' in
        `contourf_kwargs` (if given)

    contour_kwargs : dict
        keyword arguments passed to `matplotlib.pyplot.contour` (if
        `contour=True`); note: the parameters `levels` (see above) is used as
        keyword argument, i.e. it prevails over the key 'levels' in
        `contour_kwargs` (if given)

    contour_clabel_kwargs : dict
        keyword arguments passed to `matplotlib.pyplot.clabel` (if
        `contour_clabel=True`)

    interpolation : str, default: 'none'
        keyword argument 'interpolation' to be passed `matplotlib.pyplot.imshow`

    aspect : str, or scalar, default: 'equal'
        keyword argument 'aspect' to be passed `matplotlib.pyplot.imshow`

    frame : bool, default: True
        indicates if a frame is drawn around the image

    xaxis : bool, default: True
        indicates if x axis is visible

    yaxis : bool, default: True
        indicates if y axis is visible

    title : str, optional
        title of the figure

    xlabel : str, optional
        label for x axis

    ylabel : str, optional
        label for y axis

    clabel : str, optional
        label for color bar

    xticks : sequence of values, optional
        values where to place ticks along x axis

    yticks : sequence of values, optional
        values where to place ticks along y axis

    cticks : sequence of values, optional
        values where to place ticks along the color bar

    xticklabels : sequence of strs, optional,
        sequence of strings for ticks along x axis

    yticklabels : sequence of strs, optional,
        sequence of strings for ticks along y axis

    cticklabels : sequence of strs, optional,
        sequence of strings for ticks along the color bar

    xticklabels_max_decimal : int, optional
        maximal number of decimals (fractional part) for tick labels along x axis

    yticklabels_max_decimal : int, optional
        maximal number of decimals (fractional part) for tick labels along y axis

    cticklabels_max_decimal : int, optional
        maximal number of decimals (fractional part) for tick labels along the
        color bar

    colorbar_extend : str {'neither', 'both', 'min', 'max'}, default: 'neither'
        used only if `categ=False`:
        keyword argument 'extend' to be passed to `matplotlib.pyplot.colorbar`
        (or `geone.customcolors.add_colorbar`)

    colorbar_aspect : float or int, default: 20
        keyword argument 'aspect' to be passed to
        `geone.customcolors.add_colorbar`

    colorbar_pad_fraction : float or int, default: 1.0
        keyword argument 'pad_fraction' to be passed to
        `geone.customcolors.add_colorbar`

    showColorbar : bool, default: True
        indicates if the color bar (vertical) is shown

    removeColorbar : bool, default: False
        if True (and if `showColorbar=True`), then the colorbar is removed;
        note: it can be useful to draw and then remove the color bar for size of
        the plotted image...)

    showColorbarOnly : int, default: 0
        mode defining how the color bar (vertical) is shown:

        - `showColorbarOnly=0`: not used / not applied
        - `showColorbarOnly>0`: only the color bar is shown (even if \
        `showColorbar=False` or if `removeColorbar=True`):
            - `showColorbarOnly=1`: the plotted image is "cleared"
            - `showColorbarOnly=2`: an image of same color as the background is \
            drawn onto the plotted image

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        additional keyword arguments : each keyword argument with the key
        'xxx_<name>' will be passed as keyword argument with the key '<name>' to
        a function related to 'xxx';
        possibilities for 'xxx\\_' and related function:

        .. list-table::
            :widths: 25 45
            :header-rows: 1

            *   - string (prefix)
                - method from `Axes` from `matplotlib`
            *   - 'title\\_'
                - `ax.set_title()`
            *   - 'xlabel\\_'
                - `ax.set_xlabel()`
            *   - 'xticks\\_'
                - `ax.set_xticks()`
            *   - 'xticklabels\\_'
                - `ax.set_xticklabels()`
            *   - 'ylabel\\_'
                - `ax.set_ylabel()`
            *   - 'yticks\\_'
                - `ax.set_yticks()`
            *   - 'yticklabels\\_'
                - `ax.set_yticklabels()`
            *   - 'clabel\\_'
                - `cbar.set_label()`
            *   - 'cticks\\_'
                - `cbar.set_ticks()`
            *   - 'cticklabels\\_'
                - `cbar.ax.set_yticklabels()`

        for examples:

        - 'title_fontsize', '<x|y|c>label_fontsize' (keys)
        - 'title_fontweight', '<x|y|c>label_fontweight' (keys), with \
        possible values: numeric value in 0-1000 or \
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', \
        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', \
        'extra bold', 'black'

        Note that

        - default value for font size is `matplotlib.rcParams['font.size']`
        - default value for font weight is `matplotlib.rcParams['font.weight']`

    Returns
    -------
    imout : list of `matplotlib` object
        list of plotted objects
    """
    # - 'title\_'         fun: ax.set_title()
    # - 'xlabel\_'        fun: ax.set_xlabel()
    # - 'xticks\_'        fun: ax.set_xticks()
    # - 'xticklabels\_'   fun: ax.set_xticklabels()
    # - 'ylabel\_'        fun: ax.set_ylabel()
    # - 'yticks\_'        fun: ax.set_yticks()
    # - 'yticklabels\_'   fun: ax.set_yticklabels()
    # - 'clabel\_'        fun: cbar.set_label()
    # - 'cticks\_'        fun: cbar.set_ticks()
    # - 'cticklabels\_'   fun: cbar.ax.set_yticklabels()
    fname = 'drawImage2D'

    # Initialization for output
    imout = []
    #ax, cbar = None, None

    if plot_empty_grid:
        iv = None
    else:
        # Check / set iv
        if iv is None:
            if im.nv > 0:
                iv = 0
        else:
            if iv < 0:
                iv = im.nv + iv

            if iv < 0 or iv >= im.nv:
                err_msg = f'{fname}: invalid `iv` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

    # Check slice direction and indices
    n = int(ix is not None) + int(iy is not None) + int(iz is not None)

    if n == 0:
        sliceDir = 'z'
        iz = 0

    elif n==1:
        if ix is not None:
            if ix < 0:
                ix = im.nx + ix

            if ix < 0 or ix >= im.nx:
                err_msg = f'{fname}: invalid `ix` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'x'

        elif iy is not None:
            if iy < 0:
                iy = im.ny + iy

            if iy < 0 or iy >= im.ny:
                err_msg = f'{fname}: invalid `iy` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'y'

        else: # iz is not None
            if iz < 0:
                iz = im.nz + iz

            if iz < 0 or iz >= im.nz:
                err_msg = f'{fname}: invalid `iz` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'z'

    else: # n > 1
        err_msg = f'{fname}: slice specified in more than one direction'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    # Extract what to be plotted
    if sliceDir == 'x':
        dim0 = im.ny
        min0 = im.oy
        max0 = im.ymax()

        dim1 = im.nz
        min1 = im.oz
        max1 = im.zmax()
        if iv is None: # empty image
            zz = np.nan * np.ones((im.nz, im.ny))
        else:
            zz = np.array(im.val[iv, :, :, ix].reshape(dim1, dim0)) # np.array() to get a copy
                # reshape to force 2-dimensional array

    elif sliceDir == 'y':
        dim0 = im.nx
        min0 = im.ox
        max0 = im.xmax()

        dim1 = im.nz
        min1 = im.oz
        max1 = im.zmax()
        if iv is None: # empty image
            zz = np.nan * np.ones((im.nz, im.nx))
        else:
            zz = np.array(im.val[iv, :, iy, :].reshape(dim1, dim0)) # np.array() to get a copy

    else: # sliceDir == 'z'
        dim0 = im.nx
        min0 = im.ox
        max0 = im.xmax()

        dim1 = im.ny
        min1 = im.oy
        max1 = im.ymax()
        if iv is None: # empty image
            zz = np.nan * np.ones((im.ny, im.nx))
        else:
            zz = np.array(im.val[iv, iz, :, :].reshape(dim1, dim0)) # np.array() to get a copy

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

        # Get array 'dval' of displayed values
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be a 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique value in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])

        if not len(dval) and verbose > 0: # len(dval) == 0
            if logger:
                logger.warning(f'{fname}: no value to be drawn!')
            else:
                print(f'{fname}: WARNING: no value to be drawn!')

        if len(dval) > nCategMax:
            # Prevent from plotting in category mode if too many categories...
            err_msg = f'{fname}: too many categories to be plotted (> `nCategMax` = {nCategMax})'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

        # Replace dval[i] by i in zz and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            zz[zz2 == v] = i

        del zz2

        # Set 'colorList': the list of colors to use
        colorList = None
        if categCol is not None:
            if len(categCol) >= len(dval):
                colorList = [categCol[i] for i in range(len(dval))]
                # colorList = [mcolors.ColorConverter().to_rgba(categCol[i]) for i in range(len(dval))]

            elif categColCycle:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: `categCol` is used cyclically (too few entries)')
                    else:
                        print(f'{fname}: WARNING: `categCol` is used cyclically (too few entries)')
                colorList = [categCol[i%len(categCol)] for i in range(len(dval))]

            else:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: `categCol` not used (too few entries)')
                    else:
                        print(f'{fname}: WARNING: `categCol` not used (too few entries)')

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(x) for x in np.arange(len(dval)) * 1.0/(len(dval)-1)]

        # Set the colormap: 'cmap'
        if len(dval) == 1:
            cmap = ccol.custom_cmap([colorList[0], colorList[0]], ncol=2,
                                    cbad=categColbad, alpha=alpha)

        else: # len(dval) == len(colorList) > 1
            # cmap = mcolors.ListedColormap(colorList)
            cmap = ccol.custom_cmap(colorList, ncol=len(colorList),
                                    cbad=categColbad, alpha=alpha)

        # Set the min and max of the colorbar
        vmin, vmax = -0.5, len(dval) - 0.5

        # Set colorbar ticks and ticklabels if not given
        if cticks is None:
            cticks = range(len(dval))

        if cticklabels is None:
            #cticklabels = [f'{v:.3g}' for v in cticks]
            cticklabels = [f'{v:.3g}' for v in dval]

        # Reset cextend if needed
        colorbar_extend = 'neither'

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for val in np.array(excludedVal).reshape(-1): # force to be a 1d array
                np.putmask(zz, zz == val, np.nan)

    # Generate "sub-dictionaries" from kwargs
    #   For each item 'xxx_<name>':<value> from kwargs (whose the key begins
    #   with "xxx_"): the item '<name>':<value> is added to the dictionary
    #   xxx_kwargs
    #   "xxx_" is replaced by the strings below (in sub_prefix list)
    # These sub-dictionaries are passed (unpacked) as keyword arguments to
    # functions that draw labels, ticklabels, ...
    title_kwargs = {}

    xlabel_kwargs = {}
    xticks_kwargs = {}
    xticklabels_kwargs = {}

    ylabel_kwargs = {}
    yticks_kwargs = {}
    yticklabels_kwargs = {}

    clabel_kwargs = {}
    cticks_kwargs = {}
    cticklabels_kwargs = {}

    colorbar_kwargs = {'aspect':colorbar_aspect, 'pad_fraction':colorbar_pad_fraction}

    sub_prefix = ['title_',
                  'xlabel_', 'xticks_', 'xticklabels_',
                  'ylabel_', 'yticks_', 'yticklabels_',
                  'clabel_', 'cticks_', 'cticklabels_',
                  'colorbar_']
    sub_kwargs = [title_kwargs,
                  xlabel_kwargs, xticks_kwargs, xticklabels_kwargs,
                  ylabel_kwargs, yticks_kwargs, yticklabels_kwargs,
                  clabel_kwargs, cticks_kwargs, cticklabels_kwargs,
                  colorbar_kwargs] # list of dictionaries

    for k, v in kwargs.items():
        for i in range(len(sub_kwargs)):
            n = len(sub_prefix[i])
            if k[0:n] == sub_prefix[i]:
                sub_kwargs[i][k[n:]] = v # add item k[n:]:v to dictionary sub_kwargs[i]

    if showColorbarOnly:
        # Overwrite some parameters if needed
        frame = False
        xaxis = False
        yaxis = False
        title = None
        showColorbar = True
        removeColorbar = False

    # Get current axis (for plotting)
    ax = plt.gca()

    # image plot
    im_plot = ax.imshow(zz, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                        origin='lower', extent=[min0, max0, min1, max1],
                        interpolation=interpolation, aspect=aspect) #, animated=animated)
    imout.append(im_plot)

    if contourf:
        # imshow is still used above to account for 'aspect'
        # Set key word argument 'levels' from the argument 'levels'
        if contourf_kwargs is None:
            contourf_kwargs = {}
        contourf_kwargs['levels'] = levels
        im_contf = ax.contourf(zz, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                              origin='lower', extent=[min0, max0, min1, max1],
                              **contourf_kwargs)
        imout.append(im_contf)

    if contour:
        # Set key word argument 'levels' from the argument 'levels'
        contour_kwargs['levels']=levels
        im_cont = ax.contour(zz,
                              origin='lower', extent=[min0, max0, min1, max1],
                              **contour_kwargs)
        imout.append(im_cont)
        if contour_clabel:
            ax.clabel(im_cont, **contour_clabel_kwargs)

    # title
    if title is not None:
        ax.set_title(title, **title_kwargs)

    # xlabel, xticks and xticklabels
    if xlabel is not None:
        ax.set_xlabel(xlabel, **xlabel_kwargs)

    if xticks is not None:
        ax.set_xticks(xticks, **xticks_kwargs)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, **xticklabels_kwargs)
    elif xticklabels_max_decimal is not None:
        s = 10**xticklabels_max_decimal
        labels = [np.round(t*s)/s for t in ax.get_xticks()]
        ax.set_xticklabels(labels, **xticklabels_kwargs)
    elif len(xticklabels_kwargs):
        ax.set_xticklabels(ax.get_xticks(), **xticklabels_kwargs)
    # else... default xticklabels....

    # ylabel, yticks and yticklabels
    if ylabel is not None:
        ax.set_ylabel(ylabel, **ylabel_kwargs)

    if yticks is not None:
        ax.set_yticks(yticks, **yticks_kwargs)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, **yticklabels_kwargs)
    elif yticklabels_max_decimal is not None:
        s = 10**yticklabels_max_decimal
        labels = [np.round(t*s)/s for t in ax.get_yticks()]
        ax.set_yticklabels(labels, **yticklabels_kwargs)
    elif len(yticklabels_kwargs):
        ax.set_yticklabels(ax.get_yticks(), **yticklabels_kwargs)
    # else... default yticklabels....

    # Display or hide: frame, xaxis, yaxis
    if not frame:
        ax.set_frame_on(False)

    if not xaxis:
        ax.get_xaxis().set_visible(False)

    if not yaxis:
        ax.get_yaxis().set_visible(False)

    # Colorbar
    if showColorbar:
        #cbar_kwargs = {'aspect':colorbar_aspect, 'pad_fraction':colorbar_pad_fraction}
        if not contourf:
            colorbar_kwargs['extend']=colorbar_extend
        cbar = add_colorbar(im_plot, **colorbar_kwargs)

        if clabel is not None:
            cbar.set_label(clabel, **clabel_kwargs)

        if cticks is not None:
            cbar.set_ticks(cticks, **cticks_kwargs)

        if cticklabels is not None:
            cbar.ax.set_yticklabels(cticklabels, **cticklabels_kwargs)
        elif cticklabels_max_decimal is not None:
            s = 10**cticklabels_max_decimal
            labels = [np.round(t*s)/s for t in cbar.get_ticks()]
            cbar.ax.set_yticklabels(labels, **cticklabels_kwargs)
        elif len(cticklabels_kwargs):
            cbar.ax.set_yticklabels(cbar.get_ticks(), **cticklabels_kwargs)
        # else... default cticklabels....

        if removeColorbar:
            # cbar.ax.get_xaxis().set_visible(False)
            # cbar.ax.get_yaxis().set_visible(False)
            # cbar.ax.clear()
            cbar.ax.set_visible(False)

    if showColorbarOnly:
        if showColorbarOnly == 1:
            ax.clear() # change the size of the colorbar...
        else:
            # Trick: redraw the image in background color...
            zz[...] = 0
            # bg_color = mpl_rcParams['figure.facecolor'] # background color
            bg_color = ax.get_facecolor() # background color
            # bg_color = plt.gcf().get_facecolor()  # background color
            ncmap = ccol.custom_cmap([bg_color,bg_color], ncol=2)
            ax.imshow(zz, cmap=ncmap)
            ax.set_frame_on(True)
            for pos in ('bottom', 'top', 'right', 'left'):
                ax.spines[pos].set_color(bg_color)
                ax.spines[pos].set_linewidth(10)

    return imout
    #return (ax, cbar)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_colors_from_values(
        val,
        cmap=ccol.cmap_def,
        alpha=None,
        excludedVal=None,
        categ=False, categVal=None,
        categCol=None, categColCycle=False, categColbad=ccol.cbad_def,
        vmin=None, vmax=None,
        cmin=None, cmax=None,
        verbose=1,
        logger=None):
    """
    Gets the colors for given values, according to color settings as used in function :func:`drawImage2D`.

    Parameters
    ----------
    val : array-like of floats, or float
        values for which the colors have to be retrieved

    cmap : see function :func:`drawImage2D`

    alpha : see function :func:`drawImage2D`

    excludedVal : see function :func:`drawImage2D`

    categ : see function :func:`drawImage2D`

    categVal : see function :func:`drawImage2D`

    categCol : see function :func:`drawImage2D`

    categColCycle : see function :func:`drawImage2D`

    categColbad : see function :func:`drawImage2D`

    vmin : see function :func:`drawImage2D`

    vmax : see function :func:`drawImage2D`

    cmin : float, optional
        alternative keyword for `vmin` (for compatibility with color settings
        in the functions of the module `geone.imgplot3d`)

    cmax : float, optional
        alternative keyword for `vmax` (for compatibility with color settings
        in the functions of the module `geone.imgplot3d`)

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    col : 1D array of colors
        colors used for values in `val` according to the given settings
    """
    fname = 'get_colors_from_values'

    # Check vmin, cmin and vmax, cmax
    if vmin is not None and cmin is not None:
        err_msg = f'{fname}: use `vmin` or `cmin` (not both)'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    if vmax is not None and cmax is not None:
        err_msg = f'{fname}: use `vmax` or `cmax` (not both)'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    if vmin is None:
        vmin = cmin

    if vmax is None:
        vmax = cmax

    # Copy val in a 1d array
    zz = np.copy(np.atleast_1d(val)).reshape(-1)

    # --- Code adapted from function drawImage2D - start ----
    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

        # Get array 'dval' of displayed values
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be a 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique value in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])

        if not len(dval) and verbose > 0: # len(dval) == 0
            if logger:
                logger.warning(f'{fname}: no value to be drawn!')
            else:
                print(f'{fname}: WARNING: no value to be drawn!')

        # Replace dval[i] by i in zz and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            zz[zz2 == v] = i

        del zz2

        # Set 'colorList': the list of colors to use
        colorList = None
        if categCol is not None:
            if len(categCol) >= len(dval):
                colorList = [categCol[i] for i in range(len(dval))]
                # colorList = [mcolors.ColorConverter().to_rgba(categCol[i]) for i in range(len(dval))]

            elif categColCycle:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: `categCol` is used cyclically (too few entries)')
                    else:
                        print(f'{fname}: WARNING: `categCol` is used cyclically (too few entries)')
                colorList = [categCol[i%len(categCol)] for i in range(len(dval))]

            else:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: `categCol` not used (too few entries)')
                    else:
                        print(f'{fname}: WARNING: `categCol` not used (too few entries)')

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(x) for x in np.arange(len(dval)) * 1.0/(len(dval)-1)]

        # Set the colormap: 'cmap'
        if len(dval) == 1:
            cmap = ccol.custom_cmap([colorList[0], colorList[0]], ncol=2,
                                    cbad=categColbad, alpha=alpha)

        else: # len(dval) == len(colorList) > 1
            # cmap = mcolors.ListedColormap(colorList)
            cmap = ccol.custom_cmap(colorList, ncol=len(colorList),
                                    cbad=categColbad, alpha=alpha)

        # Set the min and max of the colorbar
        vmin, vmax = -0.5, len(dval) - 0.5

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for v in np.array(excludedVal).reshape(-1): # force to be a 1d array
                np.putmask(zz, zz == v, np.nan)

        if np.all(np.isnan(zz)):
            vmin, vmax= 0.0, 1.0 # any values
        else:
            if vmin is None:
                vmin = np.nanmin(zz)

            if vmax is None:
                vmax = np.nanmax(zz)

    col = cmap((zz-vmin)/(vmax-vmin))

    return col
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def drawImage2Drgb(im, nancol=(1.0, 0.0, 0.0), logger=None):
    """
    Displays a 2D image with 3 or 4 variables interpreted as RGB or RGBA code.

    Parameters
    ----------
    im : :class:`geone.img.Img`
        input image, with `im.nv=3` or `im.nv=4` variables interpreted as RGB or
        RBGA code (normalized in [0, 1])

    nancol : color, default: (1.0, 0.0, 0.0)
        color (3-tuple for RGB code, 4-tuple for RGBA code, str) used for missing
        value (`numpy.nan`) in the input image

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    fname = 'drawImage2Drgb'

    # Check image parameters
    if im.nz != 1:
        err_msg = f'{fname}: `im.nz` must be 1'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    if im.nv != 3 and im.nv != 4:
        err_msg = f'{fname}: `im.nv` must be 3 or 4'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    vv = im.val.reshape(im.nv, -1).T

    if vv.shape[1] == 3:
        nancolf = mcolors.to_rgb(nancol)
    else: # vv.shape[1] == 4
        nancolf = mcolors.to_rgba(nancol)

    ind_isnan = np.any(np.isnan(vv), axis=1)
    vv[ind_isnan, :] = nancolf

    min0, max0 = im.ox, im.xmax()
    min1, max1 = im.oy, im.ymax()

    plt.imshow(vv.reshape(im.ny, im.nx, -1), origin='lower', extent=[min0, max0, min1, max1])
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def drawGeobodyMap2D(im, iv=0, logger=None):
    """
    Displays a geobody 2D map, with adapted color bar.

    Parameters
    ----------
    im : :class:`geone.img.Img`
        input image, with variable of index `iv` interpreted as geobody labels,
        i.e.:

        - value 0: cell not in the considered medium,
        - value n > 0: cell in the n-th geobody (connected component)

    iv : int, default: 0
        index of the variable to be displayed

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    # fname = 'drawGeobodyMap2D'

    categ = True
    ngeo = int(im.val[iv].max())
    categVal = [i for i in range(1, ngeo+1)]
    categCol = None
    if ngeo <= 10:
        categCol = plt.get_cmap('tab10').colors[:ngeo]
        cticks = np.arange(ngeo)
        cticklabels = 1 + cticks
    elif ngeo <= 20:
        categCol = plt.get_cmap('tab20').colors[:ngeo]
        cticks = np.arange(ngeo)
        cticklabels = 1 + cticks
    elif ngeo <= 40:
        categCol = plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors[:ngeo-20]
        cticks = np.arange(0,ngeo,5)
        cticklabels = 1 + cticks
    else:
        categ = False
        cticks = None
        cticklabels = None
    drawImage2D(im, iv=iv, excludedVal=0, categ=categ, categVal=categVal, categCol=categCol,
                cticks=cticks, cticklabels=cticklabels, logger=logger)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeImage2Dppm(
        im,
        filename,
        ix=None, iy=None, iz=None, iv=0,
        cmap=ccol.cmap_def,
        excludedVal=None,
        categ=False, categVal=None, categCol=None,
        vmin=None, vmax=None,
        verbose=1,
        logger=None):
    """
    Writes an image in a file in ppm format.

    The colors according the to given settings, as defined in the function
    `drawImage2D` are used.

    Parameters
    ----------
    im : :class:`geone.img.Img`
        input image

    filename : str
        name of the file

    ix : see function :func:`drawImage2D`

    iy : see function :func:`drawImage2D`

    iz : see function :func:`drawImage2D`

    iv : see function :func:`drawImage2D`

    cmap : see function :func:`drawImage2D`

    excludedVal : see function :func:`drawImage2D`

    categ : see function :func:`drawImage2D`

    categVal : see function :func:`drawImage2D`

    categCol : see function :func:`drawImage2D`

    vmin : see function :func:`drawImage2D`

    vmax : see function :func:`drawImage2D`

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    """
    fname = 'writeImage2Dppm'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        err_msg = f'{fname}: invalid `iv` index'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    # Check slice direction and indices
    n = int(ix is not None) + int(iy is not None) + int(iz is not None)

    if n == 0:
        sliceDir = 'z'
        iz = 0

    elif n==1:
        if ix is not None:
            if ix < 0:
                ix = im.nx + ix

            if ix < 0 or ix >= im.nx:
                err_msg = f'{fname}: invalid `ix` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'x'

        elif iy is not None:
            if iy < 0:
                iy = im.ny + iy

            if iy < 0 or iy >= im.ny:
                err_msg = f'{fname}: invalid `iy` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'y'

        else: # iz is not None
            if iz < 0:
                iz = im.nz + iz

            if iz < 0 or iz >= im.nz:
                err_msg = f'{fname}: invalid `iz` index'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            sliceDir = 'z'

    else: # n > 1
        err_msg = f'{fname}: slice specified in more than one direction'
        if logger: logger.error(err_msg)
        raise ImgplotError(err_msg)

    # Extract what to be plotted
    if sliceDir == 'x':
        dim0 = im.ny
        min0 = im.oy
        max0 = im.ymax()

        dim1 = im.nz
        min1 = im.oz
        max1 = im.zmax()
        zz = np.array(im.val[iv, :, :, ix].reshape(dim1, dim0)) # np.array() to get a copy
            # reshape to force 2-dimensional array

    elif sliceDir == 'y':
        dim0 = im.nx
        min0 = im.ox
        max0 = im.xmax()

        dim1 = im.nz
        min1 = im.oz
        max1 = im.zmax()
        zz = np.array(im.val[iv, :, iy, :].reshape(dim1, dim0)) # np.array() to get a copy

    else: # sliceDir == 'z'
        dim0 = im.nx
        min0 = im.ox
        max0 = im.xmax()

        dim1 = im.ny
        min1 = im.oy
        max1 = im.ymax()
        zz = np.array(im.val[iv, iz, :, :].reshape(dim1, dim0)) # np.array() to get a copy

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

        # Get array 'dval' of displayed values
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be a 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise ImgplotError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique value in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])

        if not len(dval): # len(dval) == 0
            err_msg = f'{fname}: no value to be drawn' # Warning instead and not raise error...
            if logger: logger.error(err_msg)
            raise ImgplotError(err_msg)

        # Replace dval[i] by i in zz and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            zz[zz2 == v] = i

        del zz2

        # Set 'colorList': the list of colors to use
        colorList = None
        if categCol is not None:
            if len(categCol) >= len(dval):
                colorList = [categCol[i] for i in range(len(dval))]

            else:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: `categCol` not used (too few entries)')
                    else:
                        print(f'{fname}: WARNING: `categCol` not used (too few entries)')

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(0)]
            if len(dval) > 1:
                t = 1./(len(dval)-1)
                for i in range(1,len(dval)):
                    colorList.append(cmap(i*t))

        # Set the colormap: 'cmap'
        if len(dval) == 1:
            cmap = ccol.custom_cmap([colorList[0], colorList[0]], ncol=2,
                                    cbad=ccol.cbad_def)

        else: # len(dval) == len(colorList) > 1
            # cmap = mcolors.ListedColormap(colorList)
            cmap = ccol.custom_cmap(colorList, ncol=len(colorList),
                                    cbad=ccol.cbad_def)

        # Set the min and max of the colorbar
        vmin, vmax = -0.5, len(dval) - 0.5

        # Set colorbar ticks and ticklabels if not given
        if cticks is None:
            cticks = range(len(dval))

        if cticklabels is None:
            cticklabels = [f'{v:.3g}' for v in dval]

        # Reset cextend if needed
        colorbar_extend = 'neither'

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for val in np.array(excludedVal).reshape(-1): # force to be a 1d array
                np.putmask(zz, zz == val, np.nan)

    # Get dimension of zz, flip zz vertically, then reshape as a list of value
    ny = zz.shape[0]
    nx = zz.shape[1]
    zz = zz[list(reversed(range(zz.shape[0]))),:].reshape(-1)

    # Set vmin and vmax (if needed)
    if vmin is None:
        vmin = np.nanmin(zz)

    if vmax is None:
        vmax = np.nanmax(zz)

    # Get indices of bad, under, over value
    ind_bad = np.isnan(zz)
    ind_under = np.all((~ind_bad, zz < vmin),0)
    ind_over = np.all((~ind_bad, zz > vmax),0)

    # Normalize value according to colorbar
    zz = np.maximum(np.minimum((zz-vmin)/(vmax-vmin), 1.0), 0.0)

    # Get rgba code at each pixel
    rgba_arr = np.asarray([cmap(v) for v in zz])

    if cmap._rgba_bad is not None:
        rgba_arr[ind_bad] = cmap._rgba_bad

    if cmap._rgba_under is not None:
        rgba_arr[ind_under] = cmap._rgba_under

    if cmap._rgba_over is not None:
        rgba_arr[ind_over] = cmap._rgba_over

    # Convert rgb from 0-1 to integer in 0-255 (ignored a chanel)
    zz = np.round(rgba_arr[:,0:3]*255)

    # Write file (ppm)
    shead = ("P3\n"
             "# CREATED BY PYTHON3-CODE\n"
             "{0} {1}\n"
             "255\n").format(nx, ny) # header of ppm file
    # Open the file in write binary mode
    with open(filename,'wb') as ff:
        ff.write(shead.encode()) # write header
        # Write rgb values
        np.savetxt(ff, zz.reshape(-1), fmt='%d')
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.imgplot' example:")
    import matplotlib.pyplot as plt

    # Set image with 50 variables
    # ---------------------------
    # Set domain and number of cell in each direction
    xmin, xmax = -2, 2
    ymin, ymax = -1, 2
    nx, ny = 200, 150

    # Set the cell size
    sx, sy = (xmax-xmin)/nx, (ymax-ymin)/ny

    # Set the meshgrid
    x, y = xmin + 0.5 * sx + sx * np.arange(nx), ymin + 0.5 * sy + sy * np.arange(ny)
    # # equivalently:
    # x, y = np.arange(xmin+sx/2, xmax, sx), np.arange(ymin+sy/2, ymax ,sy)
    # x, y = np.linspace(xmin+sx/2, xmax-sx/2, nx), np.linspace(ymin+sy/2, ymax-sy/2, ny)
    xx,yy = np.meshgrid(x, y)

    # function values
    zz = xx**2 + yy**2 - 2

    # Set some values to nan
    zz[np.where(zz < -1.7)] = np.nan

    # set image, where each variable consists in
    # the function values 'zz' + a gaussian noise
    nv = 50
    im = Img(nx=nx, ny=ny, nz=1, nv=nv,
             sx=sx, sy=sy, sz=1.0,
             ox=xmin, oy=ymin, oz=0.0)

    for i in range(nv):
        im.set_var(ind=i, val=zz.reshape(-1)+np.random.normal(size=im.nxy()))

    # Compute the mean and standard deviation
    # ---------------------------------------
    imMean = img.imageContStat(im,op='mean')
    imStd = img.imageContStat(im,op='std',ddof=1)

    # Draw images
    # -----------
    # Set min and max value to be displayed
    vmin, vmax = -1.0, 3.0

    fig, ax = plt.subplots(2,2,figsize=(12,10))
    plt.subplot(2,2,1)
    drawImage2D(im, iv=0, vmin=vmin, vmax=vmax, title='1-st real',
                colorbar_extend='both')

    plt.subplot(2,2,2)
    drawImage2D(im, iv=1, vmin=vmin, vmax=vmax, title='2-nd real',
                colorbar_extend='both')

    plt.subplot(2,2,3)
    drawImage2D(imMean, vmin=vmin, vmax=vmax,
                title=f'Mean over {nv} real',
                colorbar_extend='both')

    plt.subplot(2,2,4)
    drawImage2D(imStd, title=f'Std over {nv} real')

    # plt.tight_layout()

    # fig.show()
    plt.show()

    # Copy im and categorize
    # ----------------------
    imCat = img.copyImg(im)
    bins = np.array([-np.inf, -1., 0., 1., np.inf])
    # set category j to a value v of image im as follows:
    #   bins[j-1] <= v < bins[j], for j=1,...,bins.size-1
    # and j=np.nan if v is np.nan

    defInd = np.array(~np.isnan(im.val)).reshape(-1) # defined indices

    imCat.val[...] = np.nan # initialization

    for j in range(1, bins.size):
        # Set category j
        imCat.val[np.all((defInd,
                          np.asarray(im.val >= bins[j-1]).reshape(-1),
                          np.asarray(im.val < bins[j]).reshape(-1)),
                         0).reshape(imCat.val.shape)] = j

    categ = list(range(1, bins.size))
    imCatProp = img.imageCategProp(imCat,categ)

    # Draw images
    # -----------
    # Set background color to "white"
    mpl_rcParams['figure.facecolor'] = 'white'

    fig, ax = plt.subplots(2,3,figsize=(18,10))
    plt.subplot(2,3,1)
    drawImage2D(imCat, iv=0, categ=True, title='1-st real')

    plt.subplot(2,3,2)
    # drawImage2D(imCat, iv=1, categ=True, title='2-nd real')
    drawImage2D(imCat, iv=1, categ=True, title='2-nd real',
                title_fontsize=18, title_fontweight='bold',
                xlabel="x-axis", xlabel_fontsize=8,
                xticks=[-2,0,2], xticklabels_fontsize=8,
                clabel="facies",clabel_fontsize=16,clabel_rotation=90,
                cticklabels=['A','B','C','D'],cticklabels_fontsize=8)

    plt.subplot(2,3,3)
    drawImage2D(imCatProp, iv=0, vmin=0, vmax=0.7, colorbar_extend='max',
                title=f'Prop. of "{categ[0]}" over {nv} real')

    plt.subplot(2,3,4)
    drawImage2D(imCatProp, iv=1, vmin=0, vmax=0.7, colorbar_extend='max',
                title=f'Prop. of "{categ[1]}" over {nv} real')

    plt.subplot(2,3,5)
    drawImage2D(imCatProp, iv=2, vmin=0, vmax=0.7, colorbar_extend='max',
                title=f'Prop. of "{categ[2]}" over {nv} real')

    plt.subplot(2,3,6)
    drawImage2D(imCatProp, iv=3, vmin=0, vmax=0.7, colorbar_extend='max',
                title=f'Prop. of "{categ[3]}" over {nv} real',
                cticks=np.arange(0,.8,.1), cticklabels=[f'{i:4.2f}' for i in np.arange(0,.8,.1)],
                cticklabels_fontweight='bold')

    plt.suptitle('Categorized images...')
    # plt.tight_layout()

    # fig.show()
    plt.show()

    a = input("Press enter to continue...")
