#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'imgplot3d.py'
# author:         Julien Straubhaar
# date:           feb-2020
# -------------------------------------------------------------------------

"""
Module for custom plots of images (class :class:`geone.img.Img`) in 3D (based on `pyvista`).
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import pyvista as pv

from geone import customcolors as ccol

# ============================================================================
class Imgplot3dError(Exception):
    """
    Custom exception related to `imgplot3d` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def drawImage3D_surface (
        im,
        plotter=None,
        ix0=0, ix1=None,
        iy0=0, iy1=None,
        iz0=0, iz1=None,
        iv=0,
        cmap=ccol.cmap_def,
        cmin=None, cmax=None,
        alpha=None,
        excludedVal=None,
        categ=False,
        ncateg_max=30,
        categVal=None,
        categCol=None,
        categColCycle=False,
        categActive=None,
        use_clip_plane=False,
        show_scalar_bar=True,
        show_outline=True,
        show_bounds=False,
        show_axes=True,
        text=None,
        scalar_bar_annotations=None,
        scalar_bar_annotations_max=20,
        scalar_bar_kwargs=None,
        outline_kwargs=None,
        bounds_kwargs=None,
        axes_kwargs=None,
        text_kwargs=None,
        background_color=None,
        foreground_color=None,
        cpos=None,
        verbose=1,
        logger=None,
        **kwargs):
    """
    Displays a 3D image as surface(s) (based on `pyvista`).

    Parameters
    ----------
    im : :class:`geone.img.Img`
        image (3D)

    plotter : :class:`pyvista.Plotter`, optional
        - if given (not `None`), add element to the plotter, a further call to \
        `plotter.show()` will be required to show the plot
        - if not given (`None`, default): a plotter is created and the plot \
        is shown

    ix0 : int, default: 0
        index of first slice along x direction, considered for plotting

    ix1 : int, optional
        1+index of last slice along x direction (`ix0 < ix1`), considered for
        plotting; by default: number of cells in x direction (`ix1=im.nx`) is
        used

    iy0 : int, default: 0
        index of first slice along y direction, considered for plotting

    iy1 : int, optional
        1+index of last slice along y direction (`iy0 < iy1`), considered for
        plotting; by default: number of cells in x direction (`iy1=im.ny`) is
        used

    iz0 : int, default: 0
        index of first slice along z direction, considered for plotting

    iz1 : int, optional
        1+index of last slice along z direction (`iz0 < iz1`), considered for
        plotting; by default: number of cells in z direction (`iz1=im.nz`) is
        used

    iv : int, default: 0
        index of the variable to be displayed

    cmap : colormap, default: `geone.customcolors.cmap_def`
        color map (can be a string, in this case the color map is obtained by
        `matplotlib.pyplot.get_cmap(cmap)`)

    cmin : float, optional
        used only if `categ=False`:
        minimal value to be displayed; by default: minimal value of the displayed
        variable is used for `cmin`

    cmax : float, optional
        used only if `categ=False`:
        maximal value to be displayed; by default: maximal value of the displayed
        variable is used for `cmax`

    alpha : float, optional
        value of the "alpha" channel (for transparency);
        by default (`None`): `alpha=1.0` is used (no transparency)

    excludedVal : sequence of values, or single value, optional
        values to be excluded from the plot;
        note not used if `categ=True` and `categVal` is not `None`

    categ : bool, default: False
        indicates if the variable of the image to diplay has to be treated as a
        categorical (discrete) variable (True), or continuous variable (False)

    ncateg_max : int, default: 30
        used only if `categ=True`:
        maximal number of categories, if there are more category values and
        `categVal=None`, nothing is plotted (`categ` should set to False)

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

    categActive : 1D array-like of bools, optional
        used only if `categ=True`, sequence of same length as `categVal`:

        - `categActive[i]=True`: `categVal[i]` is displayed
        - `categActive[i]=False`: `categVal[i]` is not displayed

        by default (`None`): all category values `categVal` are displayed

    use_clip_plane : bool, default: False
        if `True`: the function `pyvista.add_mesh_clip_plane` (allowing
        interactive clipping) is used instead of `pyvista.add_mesh`

    show_scalar_bar : bool, default: True
        indicates if scalar bar (color bar) is displayed

    show_outline : bool, default: True
        indicates if outline (around the image) is displayed

    show_bounds : bool, default: False
        indicates if bounds are displayed (box with graduation)

    show_axes : bool, default: True
        indicates if axes are displayed

    text : str, optional
        text (title) to be displayed on the figure

    scalar_bar_annotations : dict, optional
        annotation (ticks) on the scalar bar (color bar), used if
        `show_scalar_bar=True`

    scalar_bar_annotations_max : int, default: 20
        maximal number of annotations (ticks) on the scalar bar (color bar)
        when `categ=True` and `scalar_bar_annotations=None`

    scalar_bar_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar`
        (can be useful for customization, used if `show_scalar_bar=True`)
        note: in subplots (multi-sub-window), key 'title' should be distinct for
        each subplot

    outline_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_mesh`
        (can be useful for customization, used if `show_outline=True`)

    bounds_kwargs : dict, optional
        keyword arguments passed to function `plotter.show_bounds`
        (can be useful for customization, used if `show_bounds=True`)

    axes_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_axes`
        (can be useful for customization, used if `show_axes=True`)

    text_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_text`
        (can be useful for customization, used if `text` is not `None`)

    background_color : color, optional
        background color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    foreground_color : color, optional
        foreground color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        additional keyword arguments passed to `plotter.add_mesh[_clip_plane]`
        when plotting the variable, such as

        - opacity (float, or str) : \
        opacity for colors; \
        default: 'linear', (set 'linear_r' to invert opacity)
        - show_edges (bool) : \
        indicates if edges of the grid are displayed
        - edge_color (color) : \
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str) for edges \
        (used if `show_edges=True`)
        - line_width (float) \
        line width for edges (used if `show_edges=True`)
        - etc.

    Notes
    -----
    - 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """
    fname = 'drawImage3D_surface'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        err_msg = f'{fname}: invalid `iv` index'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        err_msg = f'{fname}: invalid indices along x axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        err_msg = f'{fname}: invalid indices along y axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        err_msg = f'{fname}: invalid indices along z axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

    # Initialization of dictionary (do not use {} as default argument, it is not re-initialized...)
    if scalar_bar_annotations is None:
        scalar_bar_annotations = {}

    if scalar_bar_kwargs is None:
        scalar_bar_kwargs = {}

    if outline_kwargs is None:
        outline_kwargs = {}

    if bounds_kwargs is None:
        bounds_kwargs = {}

    if axes_kwargs is None:
        axes_kwargs = {}

    if text_kwargs is None:
        text_kwargs = {}

    # Extract what to be plotted
    # zz = np.array(im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1]) # np.array() to get a copy
    zz = im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1].flatten() # .flatten() provides a copy

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                err_msg = f'{fname}: too many categories, set `categ=False`'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        if not len(dval): # len(dval) == 0
            err_msg = f'{fname}: no value to be drawn'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        if categActive is not None:
            if len(categActive) != len(dval):
                err_msg = f'{fname}: length of `categActive` invalid (should be the same as length of `categVal`)'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        else:
            categActive = np.ones(len(dval), dtype='bool')

        # Replace dval[i] by i in zz if categActive[i] is True otherwise by np.nan, and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            if categActive[i]:
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
        # - Trick: duplicate last color (if len(colorList)> 1)!
        if len(colorList) == 1:
            colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5] = f'{v:.3g}'

        scalar_bar_kwargs['n_labels'] = 0
        scalar_bar_kwargs['n_colors'] = len(dval)

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for val in np.array(excludedVal).reshape(-1): # force to be an 1d array
                np.putmask(zz, zz == val, np.nan)

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    # Set pyvista ImageData (previously: UniformGrid)
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    # pg = pv.UniformGrid(dims=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    # pg = pv.UniformGrid(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    pg = pv.ImageData(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))

    pg.cell_data[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    if use_clip_plane:
        add_mesh_func = pp.add_mesh_clip_plane
    else:
        add_mesh_func = pp.add_mesh

    add_mesh_func(pg.threshold(value=(cmin, cmax)), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        # # - old -
        # # pg.cell_arrays[im.varname[iv]][...] = np.nan # trick: set all value to nan and use nan_opacity = 0 for empty plot but 'saving' the scalar bar...
        # # # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=True, scalar_bar_args=scalar_bar_kwargs)
        # # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
        # # - old -
        # # Trick: set opacity=0 and nan_opacity=0 for empty plot but 'saving' the scalar bar...
        # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), opacity=0., nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
        if 'title' not in scalar_bar_kwargs.keys():
            scalar_bar_kwargs['title'] = im.varname[iv]
        pp.add_scalar_bar(**scalar_bar_kwargs)

    if show_outline:
        pp.add_mesh(pg.outline(), **outline_kwargs)

    if show_bounds:
        pp.show_bounds(**bounds_kwargs)

    if show_axes:
        pp.add_axes(**axes_kwargs)

    if text is not None:
        pp.add_text(text, **text_kwargs)

    if plotter is None:
        pp.show(cpos=cpos)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def drawImage3D_slice (
        im,
        plotter=None,
        ix0=0, ix1=None,
        iy0=0, iy1=None,
        iz0=0, iz1=None,
        iv=0,
        slice_normal_x=None,
        slice_normal_y=None,
        slice_normal_z=None,
        slice_normal_custom=None,
        cmap=ccol.cmap_def,
        cmin=None, cmax=None,
        alpha=None,
        excludedVal=None,
        categ=False,
        ncateg_max=30,
        categVal=None,
        categCol=None,
        categColCycle=False,
        categActive=None,
        show_scalar_bar=True,
        show_outline=True,
        show_bounds=False,
        show_axes=True,
        text=None,
        scalar_bar_annotations=None,
        scalar_bar_annotations_max=20,
        scalar_bar_kwargs=None,
        outline_kwargs=None,
        bounds_kwargs=None,
        axes_kwargs=None,
        text_kwargs=None,
        background_color=None,
        foreground_color=None,
        cpos=None,
        verbose=1,
        logger=None,
        **kwargs):
    """
    Displays a 3D image as slices(s) (based on `pyvista`).

    Parameters
    ----------
    im : :class:`geone.img.Img`
        image (3D)

    plotter : :class:`pyvista.Plotter`, optional
        - if given (not `None`), add element to the plotter, a further call to \
        `plotter.show()` will be required to show the plot
        - if not given (`None`, default): a plotter is created and the plot \
        is shown

    ix0 : int, default: 0
        index of first slice along x direction, considered for plotting

    ix1 : int, optional
        1+index of last slice along x direction (`ix0 < ix1`), considered for
        plotting; by default: number of cells in x direction (`ix1=im.nx`) is
        used

    iy0 : int, default: 0
        index of first slice along y direction, considered for plotting

    iy1 : int, optional
        1+index of last slice along y direction (`iy0 < iy1`), considered for
        plotting; by default: number of cells in x direction (`iy1=im.ny`) is
        used

    iz0 : int, default: 0
        index of first slice along z direction, considered for plotting

    iz1 : int, optional
        1+index of last slice along z direction (`iz0 < iz1`), considered for
        plotting; by default: number of cells in z direction (`iz1=im.nz`) is
        used

    iv : int, default: 0
        index of the variable to be displayed

    slice_normal_x : sequence of values, or single value, optional
        values of the (float) x coordinate where a slice normal to x axis is
        displayed

    slice_normal_y : sequence of values, or single value, optional
        values of the (float) y coordinate where a slice normal to y axis is
        displayed

    slice_normal_z : sequence of values, or single value, optional
        values of the (float) z coordinate where a slice normal to z axis is
        displayed

    slice_normal_custom : (sequence of) sequence(s) of two 3-tuple, optional
        definition of custom normal slice(s) to be displayed, a slice is
        defined by a sequence two 3-tuple, ((vx, vy, vz), (px, py, pz)): slice
        normal to the vector (vx, vy, vz) and going through the point (px, py, pz)

    cmap : colormap, default: `geone.customcolors.cmap_def`
        color map (can be a string, in this case the color map
        `matplotlib.pyplot.get_cmap(cmap)`)

    cmin : float, optional
        used only if `categ=False`:
        minimal value to be displayed; by default: minimal value of the displayed
        variable is used for `cmin`

    cmax : float, optional
        used only if `categ=False`:
        maximal value to be displayed; by default: maximal value of the displayed
        variable is used for `cmax`

    alpha : float, optional
        value of the "alpha" channel (for transparency);
        by default (`None`): `alpha=1.0` is used (no transparency)

    excludedVal : sequence of values, or single value, optional
        values to be excluded from the plot;
        note not used if `categ=True` and `categVal` is not `None`

    categ : bool, default: False
        indicates if the variable of the image to diplay has to be treated as a
        categorical (discrete) variable (True), or continuous variable (False)

    ncateg_max : int, default: 30
        used only if `categ=True`:
        maximal number of categories, if there are more category values and
        `categVal=None`, nothing is plotted (`categ` should set to False)

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

    categActive : 1D array-like of bools, optional
        used only if `categ=True`, sequence of same length as `categVal`:

        - `categActive[i]=True`: `categVal[i]` is displayed
        - `categActive[i]=False`: `categVal[i]` is not displayed

        by default (`None`): all category values `categVal` are displayed

    show_scalar_bar : bool, default: True
        indicates if scalar bar (color bar) is displayed

    show_outline : bool, default: True
        indicates if outline (around the image) is displayed

    show_bounds : bool, default: False
        indicates if bounds are displayed (box with graduation)

    show_axes : bool, default: True
        indicates if axes are displayed

    text : str, optional
        text (title) to be displayed on the figure

    scalar_bar_annotations : dict, optional
        annotation (ticks) on the scalar bar (color bar), used if
        `show_scalar_bar=True`
    scalar_bar_annotations_max : int, default: 20
        maximal number of annotations (ticks) on the scalar bar (color bar)
        when `categ=True` and `scalar_bar_annotations=None`

    scalar_bar_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar`
        (can be useful for customization, used if `show_scalar_bar=True`)
        note: in subplots (multi-sub-window), key 'title' should be distinct for
        each subplot

    outline_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_mesh`
        (can be useful for customization, used if `show_outline=True`)

    bounds_kwargs : dict, optional
        keyword arguments passed to function `plotter.show_bounds`
        (can be useful for customization, used if `show_bounds=True`)

    axes_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_axes`
        (can be useful for customization, used if `show_axes=True`)

    text_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_text`
        (can be useful for customization, used if `text` is not `None`)

    background_color : color, optional
        background color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    foreground_color : color, optional
        foreground color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        additional keyword arguments passed to `plotter.add_mesh`
        when plotting the variable, such as

        - opacity (float, or str) : \
        opacity for colors; \
        default: 'linear', (set 'linear_r' to invert opacity)
        - show_edges (bool) : \
        indicates if edges of the grid are displayed
        - edge_color (color) : \
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str) for edges \
        (used if `show_edges=True`)
        - line_width (float) \
        line width for edges (used if `show_edges=True`)
        - etc.

    Notes
    -----
    - 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """
    fname = 'drawImage3D_slice'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        err_msg = f'{fname}: invalid `iv` index'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        err_msg = f'{fname}: invalid indices along x axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        err_msg = f'{fname}: invalid indices along y axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        err_msg = f'{fname}: invalid indices along z axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

    # Initialization of dictionary (do not use {} as default argument, it is not re-initialized...)
    if scalar_bar_annotations is None:
        scalar_bar_annotations = {}

    if scalar_bar_kwargs is None:
        scalar_bar_kwargs = {}

    if outline_kwargs is None:
        outline_kwargs = {}

    if bounds_kwargs is None:
        bounds_kwargs = {}

    if axes_kwargs is None:
        axes_kwargs = {}

    if text_kwargs is None:
        text_kwargs = {}

    # Extract what to be plotted
    # zz = np.array(im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1]) # np.array() to get a copy
    zz = im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1].flatten() # .flatten() provides a copy

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                err_msg = f'{fname}: too many categories, set `categ=False`'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        if not len(dval): # len(dval) == 0
            err_msg = f'{fname}: no value to be drawn'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        if categActive is not None:
            if len(categActive) != len(dval):
                err_msg = f'{fname}: length of `categActive` invalid (should be the same as length of `categVal`)'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        else:
            categActive = np.ones(len(dval), dtype='bool')

        # Replace dval[i] by i in zz if categActive[i] is True otherwise by np.nan, and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            if categActive[i]:
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
        # - Trick: duplicate last color (if len(colorList)> 1)!
        if len(colorList) == 1:
            colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5] = f'{v:.3g}'

        scalar_bar_kwargs['n_labels'] = 0
        scalar_bar_kwargs['n_colors'] = len(dval)

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for val in np.array(excludedVal).reshape(-1): # force to be an 1d array
                np.putmask(zz, zz == val, np.nan)

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    # Set pyvista ImageData (previously: UniformGrid)
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    # pg = pv.UniformGrid(dims=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    # pg = pv.UniformGrid(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    pg = pv.ImageData(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))

    pg.cell_data[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    if slice_normal_x is not None:
        for v in np.array(slice_normal_x).reshape(-1):
            pp.add_mesh(pg.slice(normal=(1,0,0), origin=(v,0,0)), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if slice_normal_y is not None:
        for v in np.array(slice_normal_y).reshape(-1):
            pp.add_mesh(pg.slice(normal=(0,1,0), origin=(0,v,0)), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if slice_normal_z is not None:
        for v in np.array(slice_normal_z).reshape(-1):
            pp.add_mesh(pg.slice(normal=(0,0,1), origin=(0,0,v)), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if slice_normal_custom is not None:
        for nor, ori in np.array(slice_normal_custom).reshape(-1, 2, 3):
            pp.add_mesh(pg.slice(normal=nor, origin=ori), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        # # - old -
        # # pg.cell_arrays[im.varname[iv]][...] = np.nan # trick: set all value to nan and use nan_opacity = 0 for empty plot but 'saving' the scalar bar...
        # # # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=True, scalar_bar_args=scalar_bar_kwargs)
        # # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
        # # - old -
        # # Trick: set opacity=0 and nan_opacity=0 for empty plot but 'saving' the scalar bar...
        # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), opacity=0., nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
        if 'title' not in scalar_bar_kwargs.keys():
            scalar_bar_kwargs['title'] = im.varname[iv]
        pp.add_scalar_bar(**scalar_bar_kwargs)

    if show_outline:
        pp.add_mesh(pg.outline(), **outline_kwargs)

    if show_bounds:
        pp.show_bounds(**bounds_kwargs)

    if show_axes:
        pp.add_axes(**axes_kwargs)

    if text is not None:
        pp.add_text(text, **text_kwargs)

    if plotter is None:
        pp.show(cpos=cpos)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def drawImage3D_empty_grid (
        im,
        plotter=None,
        ix0=0, ix1=None,
        iy0=0, iy1=None,
        iz0=0, iz1=None,
        cmap=ccol.cmap_def,
        cmin=None, cmax=None,
        alpha=None,
        excludedVal=None,
        categ=False,
        ncateg_max=30,
        categVal=None,
        categCol=None,
        categColCycle=False,
        categActive=None,
        show_scalar_bar=True,
        show_outline=True,
        show_bounds=False,
        show_axes=True,
        text=None,
        scalar_bar_annotations=None,
        scalar_bar_annotations_max=20,
        scalar_bar_kwargs=None,
        outline_kwargs=None,
        bounds_kwargs=None,
        axes_kwargs=None,
        text_kwargs=None,
        background_color=None,
        foreground_color=None,
        cpos=None,
        verbose=1,
        logger=None,
        **kwargs):
    """
    Displays an empty grid from a 3D image.

    Same parameters (if present) as in function :func:`drawImage3D_slice` are used,
    see this function. (Tricks are applied.)
    """
    fname = 'drawImage3D_empty_grid'

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        err_msg = f'{fname}: invalid indices along x axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        err_msg = f'{fname}: invalid indices along y axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        err_msg = f'{fname}: invalid indices along z axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

    # Initialization of dictionary (do not use {} as default argument, it is not re-initialized...)
    if scalar_bar_annotations is None:
        scalar_bar_annotations = {}

    if scalar_bar_kwargs is None:
        scalar_bar_kwargs = {}

    if outline_kwargs is None:
        outline_kwargs = {}

    if bounds_kwargs is None:
        bounds_kwargs = {}

    if axes_kwargs is None:
        axes_kwargs = {}

    if text_kwargs is None:
        text_kwargs = {}

    # Extract what to be plotted
    vname = 'tmp'
    zz = np.nan * np.ones((iz1-iz0, iy1-iy0, ix1-ix0)).flatten()

    if categ:
        # --- Treat categorical variable ---
        if categCol is not None \
                and type(categCol) is not list \
                and type(categCol) is not tuple:
            err_msg = f'{fname}: `categCol` must be a list or a tuple (if not `None`)'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                err_msg = f'{fname}: `categVal` contains duplicated entries'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                err_msg = f'{fname}: length of `categVal` and length of `categCol` differ'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                err_msg = f'{fname}: too many categories, set `categ=False`'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)

        if not len(dval): # len(dval) == 0
            err_msg = f'{fname}: no value to be drawn'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

        if categActive is not None:
            if len(categActive) != len(dval):
                err_msg = f'{fname}: length of `categActive` invalid (should be the same as length of `categVal`)'
                if logger: logger.error(err_msg)
                raise Imgplot3dError(err_msg)
        else:
            categActive = np.ones(len(dval), dtype='bool')

        # Replace dval[i] by i in zz if categActive[i] is True otherwise by np.nan, and other values by np.nan
        zz2 = np.array(zz) # copy array
        zz[...] = np.nan # initialize
        for i, v in enumerate(dval):
            if categActive[i]:
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
        # - Trick: duplicate last color (if len(colorList)> 1)!
        if len(colorList) == 1:
            colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5] = f'{v:.3g}'

        scalar_bar_kwargs['n_labels'] = 0
        scalar_bar_kwargs['n_colors'] = len(dval)

    else: # categ == False
        # --- Treat continuous variable ---
        # Possibly exclude values from zz
        if excludedVal is not None:
            for val in np.array(excludedVal).reshape(-1): # force to be an 1d array
                np.putmask(zz, zz == val, np.nan)

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    # Set pyvista ImageData (previously: UniformGrid)
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    # pg = pv.UniformGrid(dims=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    # pg = pv.UniformGrid(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    pg = pv.ImageData(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))

    pg.cell_data[vname] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    # Here is the trick!
    kwargs['opacity']=0.0
    pp.add_mesh(pg.slice(normal=(1,0,0), origin=(im.ox + (ix0+0.5)*im.sx,0,0)), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        pp.add_scalar_bar(**scalar_bar_kwargs)

    if show_outline:
        pp.add_mesh(pg.outline(), **outline_kwargs)

    if show_bounds:
        pp.show_bounds(**bounds_kwargs)

    if show_axes:
        pp.add_axes(**axes_kwargs)

    if text is not None:
        pp.add_text(text, **text_kwargs)

    if plotter is None:
        pp.show(cpos=cpos)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def drawImage3D_volume (
        im,
        plotter=None,
        ix0=0, ix1=None,
        iy0=0, iy1=None,
        iz0=0, iz1=None,
        iv=0,
        cmap=ccol.cmap_def,
        cmin=None, cmax=None,
        set_out_values_to_nan=True,
        show_scalar_bar=True,
        show_outline=True,
        show_bounds=False,
        show_axes=True,
        text=None,
        scalar_bar_annotations=None,
        scalar_bar_kwargs=None,
        outline_kwargs=None,
        bounds_kwargs=None,
        axes_kwargs=None,
        text_kwargs=None,
        background_color=None,
        foreground_color=None,
        cpos=None,
        logger=None,
        **kwargs):
    """
    Displays a 3D image as volume (based on `pyvista`).

    Parameters
    ----------
    im : :class:`geone.img.Img`
        image (3D)

    plotter : :class:`pyvista.Plotter`, optional
        - if given (not `None`), add element to the plotter, a further call to \
        `plotter.show()` will be required to show the plot
        - if not given (`None`, default): a plotter is created and the plot \
        is shown

    ix0 : int, default: 0
        index of first slice along x direction, considered for plotting

    ix1 : int, optional
        1+index of last slice along x direction (`ix0 < ix1`), considered for
        plotting; by default: number of cells in x direction (`ix1=im.nx`) is
        used

    iy0 : int, default: 0
        index of first slice along y direction, considered for plotting

    iy1 : int, optional
        1+index of last slice along y direction (`iy0 < iy1`), considered for
        plotting; by default: number of cells in x direction (`iy1=im.ny`) is
        used

    iz0 : int, default: 0
        index of first slice along z direction, considered for plotting

    iz1 : int, optional
        1+index of last slice along z direction (`iz0 < iz1`), considered for
        plotting; by default: number of cells in z direction (`iz1=im.nz`) is
        used

    iv : int, default: 0
        index of the variable to be displayed

    cmap : colormap, default: `geone.customcolors.cmap_def`
        color map (can be a string, in this case the color map is obtained by
        `matplotlib.pyplot.get_cmap(cmap)`)

    cmin : float, optional
        used only if `categ=False`:
        minimal value to be displayed; by default: minimal value of the displayed
        variable is used for `cmin`

    cmax : float, optional
        used only if `categ=False`:
        maximal value to be displayed; by default: maximal value of the displayed
        variable is used for `cmax`

    set_out_values_to_nan : bool, default: True
        indicates if values out of the range `[cmin, cmax]` is set to `numpy.nan`
        before plotting

    show_scalar_bar : bool, default: True
        indicates if scalar bar (color bar) is displayed

    show_outline : bool, default: True
        indicates if outline (around the image) is displayed

    show_bounds : bool, default: False
        indicates if bounds are displayed (box with graduation)

    show_axes : bool, default: True
        indicates if axes are displayed

    text : str, optional
        text (title) to be displayed on the figure

    scalar_bar_annotations : dict, optional
        annotation (ticks) on the scalar bar (color bar), used if
        `show_scalar_bar=True`

    scalar_bar_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar`
        (can be useful for customization, used if `show_scalar_bar=True`)
        note: in subplots (multi-sub-window), key 'title' should be distinct for
        each subplot

    outline_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_mesh`
        (can be useful for customization, used if `show_outline=True`)

    bounds_kwargs : dict, optional
        keyword arguments passed to function `plotter.show_bounds`
        (can be useful for customization, used if `show_bounds=True`)

    axes_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_axes`
        (can be useful for customization, used if `show_axes=True`)

    text_kwargs : dict, optional
        keyword arguments passed to function `plotter.add_text`
        (can be useful for customization, used if `text` is not `None`)

    background_color : color, optional
        background color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    foreground_color : color, optional
        foreground color (3-tuple (RGB code), 4-tuple (RGBA code) or str)

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        additional keyword arguments passed to `plotter.add_volume`
        when plotting the variable, such as

        - opacity (float, or str) : \
        opacity for colors; \
        default: 'linear', (set 'linear_r' to invert opacity)
        - show_edges (bool) : \
        indicates if edges of the grid are displayed
        - edge_color (color) : \
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str) for edges \
        (used if `show_edges=True`)
        - line_width (float) \
        line width for edges (used if `show_edges=True`)
        - etc.

    Notes
    -----
    - 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """
    fname = 'drawImage3D_volume'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        err_msg = f'{fname}: invalid `iv` index'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        err_msg = f'{fname}: invalid indices along x axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        err_msg = f'{fname}: invalid indices along y axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        err_msg = f'{fname}: invalid indices along z axis'
        if logger: logger.error(err_msg)
        raise Imgplot3dError(err_msg)

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            err_msg = f'{fname}: invalid `cmap` string'
            if logger: logger.error(err_msg)
            raise Imgplot3dError(err_msg)

    # Initialization of dictionary (do not use {} as default argument, it is not re-initialized...)
    if scalar_bar_annotations is None:
        scalar_bar_annotations = {}

    if scalar_bar_kwargs is None:
        scalar_bar_kwargs = {}

    if outline_kwargs is None:
        outline_kwargs = {}

    if bounds_kwargs is None:
        bounds_kwargs = {}

    if axes_kwargs is None:
        axes_kwargs = {}

    if text_kwargs is None:
        text_kwargs = {}

    # Extract what to be plotted
    # zz = np.array(im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1]) # np.array() to get a copy
    zz = im.val[iv][iz0:iz1, iy0:iy1, ix0:ix1].flatten() # .flatten() provides a copy

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    if set_out_values_to_nan:
        np.putmask(zz, np.any((np.isnan(zz), zz < cmin, zz > cmax), axis=0), np.nan)

    # Set pyvista ImageData (previously: UniformGrid)
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    # pg = pv.UniformGrid(dims=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    # pg = pv.UniformGrid(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))
    pg = pv.ImageData(dimensions=(xdim, ydim, zdim), spacing=(im.sx, im.sy, im.sz), origin=(xmin, ymin, zmin))

    pg.cell_data[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    # pp.add_volume(pg.ctp(), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=show_scalar_bar, scalar_bar_args=scalar_bar_kwargs)
    # pp.add_volume(pg.ctp(), cmap=cmap, clim=(cmin, cmax), opacity_unit_distance=0.1, opacity=opacity, annotations=scalar_bar_annotations, show_scalar_bar=False)
    pp.add_volume(pg.ctp(), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False, **kwargs)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), opacity=0., nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
        if 'title' not in scalar_bar_kwargs.keys():
            scalar_bar_kwargs['title'] = im.varname[iv]
        pp.add_scalar_bar(**scalar_bar_kwargs)

    if show_outline:
        pp.add_mesh(pg.outline(), **outline_kwargs)

    if show_bounds:
        pp.show_bounds(**bounds_kwargs)

    if show_axes:
        pp.add_axes(**axes_kwargs)

    if text is not None:
        pp.add_text(text, **text_kwargs)

    if plotter is None:
        pp.show(cpos=cpos)
# ----------------------------------------------------------------------------

# From: https://docs.pyvista.org/plotting/plotting.html?highlight=add_mesh#pyvista.BasePlotter.add_mesh
# add_scalar_bar(title=None, n_labels=5, italic=False, bold=False, title_font_size=None, label_font_size=None, color=None, font_family=None, shadow=False, mapper=None, width=None, height=None, position_x=None, position_y=None, vertical=None, interactive=None, fmt=None, use_opacity=True, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False, render=True)
#
#     Create scalar bar using the ranges as set by the last input mesh.
#
#     Parameters
#
#             title (string, optional) – Title of the scalar bar. Default None
#
#             n_labels (int, optional) – Number of labels to use for the scalar bar.
#
#             italic (bool, optional) – Italicises title and bar labels. Default False.
#
#             bold (bool, optional) – Bolds title and bar labels. Default True
#
#             title_font_size (float, optional) – Sets the size of the title font. Defaults to None and is sized automatically.
#
#             label_font_size (float, optional) – Sets the size of the title font. Defaults to None and is sized automatically.
#
#             color (string or 3 item list, optional, defaults to white) –
#
#             Either a string, rgb list, or hex color string. For example:
#
#                 color=’white’ color=’w’ color=[1, 1, 1] color=’#FFFFFF’
#
#             font_family (string, optional) – Font family. Must be either courier, times, or arial.
#
#             shadow (bool, optional) – Adds a black shadow to the text. Defaults to False
#
#             width (float, optional) – The percentage (0 to 1) width of the window for the colorbar
#
#             height (float, optional) – The percentage (0 to 1) height of the window for the colorbar
#
#             position_x (float, optional) – The percentage (0 to 1) along the windows’s horizontal direction to place the bottom left corner of the colorbar
#
#             position_y (float, optional) – The percentage (0 to 1) along the windows’s vertical direction to place the bottom left corner of the colorbar
#
#             interactive (bool, optional) – Use a widget to control the size and location of the scalar bar.
#
#             use_opacity (bool, optional) – Optionally display the opacity mapping on the scalar bar
#
#             outline (bool, optional) – Optionally outline the scalar bar to make opacity mappings more obvious.
#
#             nan_annotation (bool, optional) – Annotate the NaN color
#
#             below_label (str, optional) – String annotation for values below the scalars range
#
#             above_label (str, optional) – String annotation for values above the scalars range
#
#             background_color (array, optional) – The color used for the background in RGB format.
#
#             n_colors (int, optional) – The maximum number of color displayed in the scalar bar.
#
#             fill (bool) – Draw a filled box behind the scalar bar with the background_color
#
#             render (bool, optional) – Force a render when True. Default True.
#
#     Notes
#
#     Setting title_font_size, or label_font_size disables automatic font sizing for both the title and label.

if __name__ == "__main__":
    print("Module 'geone.imgplot3d' example:")

    # Example with a 3D gaussian random field

    import geone.covModel as gcm
    from geone import grf, img

    # Define grid
    nx, ny, nz = 85, 56, 34
    dx, dy, dz = 1., 1., 1.
    ox, oy, oz = 0., 0., 0.

    dimension = [nx, ny, nz]
    spacing = [dx, dy, dy]
    origin = [ox, oy, oz]

    # Define covariance model
    cov_model = gcm.CovModel3D(elem=[
                    ('gaussian', {'w':8.9, 'r':[40, 20, 10]}), # elementary contribution
                    ('nugget', {'w':0.1})                      # elementary contribution
                    ], alpha=-30, beta=-45, gamma=20, name='')

    # Set seed
    np.random.seed(123)

    # Generate GRF
    v = grf.grf3D(cov_model, (nx, ny, nz), (dx, dy, dz), (ox, oy, oz))
    im = img.Img(nx, ny, nz, dx, dy, dz, ox, oy, oz, nv=1, val=v)

    # ===== Ex1 =====
    # Simple plot
    # ------
    drawImage3D_volume(im, text='Ex1: volume (cont.)')

    # # Equivalent:
    # pp = pv.Plotter()
    # drawImage3D_volume(im, text='Ex1: volume (cont.)')
    # pp.show()

    # # For saving screenshot (png)
    # # Note: axes will not be displayed in off screen mode
    # pp = pv.Plotter(off_screen=True)
    # drawImage3D_volume(im, plotter=pp)
    # pp.show(screenshot='test.png')

    # ===== Ex2 =====
    # Multiple plot
    # ------
    # Note: scalar bar is not displayed in all plots (even if show_scalar_bar is True) when
    #       same title for scalar bar is used (see Ex4)
    pp = pv.Plotter(shape=(2,2))

    pp.subplot(0,0)
    drawImage3D_surface(im, plotter=pp, text='Ex2: surface (cont.)' )

    pp.subplot(0,1)
    drawImage3D_volume(im, plotter=pp, text='Ex2: volume (cont.)')

    cx, cy, cz = im.ox+0.5*im.nx*im.sx, im.oy+0.5*im.ny*im.sy, im.oz+0.5*im.nz*im.sz # center of image
    pp.subplot(1,0)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text='Ex2: slice (cont.)')

    pp.subplot(1,1)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_custom=[[(1, 1, 0), (cx, cy, cz)], [(1, -1, 0), (cx, cy, cz)]],
        text='Ex2: slice (cont.)')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # ===== Ex3 =====
    # Multiple plot
    # ------
    # Note: scalar bar is not displayed in all plots (even if show_scalar_bar is True) when
    #       same title for scalar bar is used (see Ex4)
    pp = pv.Plotter(shape=(1,3))

    pp.subplot(0,0)
    drawImage3D_volume(im, plotter=pp, cmin=2, cmax=4, text='Ex3: volume - cmin / cmax')

    pp.subplot(0,1)
    drawImage3D_volume(im, plotter=pp, cmin=2, cmax=4, set_out_values_to_nan=False, text='Ex3: volume - cmin / cmax - set_out_values_to_nan=False')

    pp.subplot(0,2)
    drawImage3D_surface(im, plotter=pp, cmin=2, cmax=4, text='Ex3: surface - cmin / cmax')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # Categorize image...
    # -------------------
    v = im.val.reshape(-1)
    newv = np.zeros(im.nxyz())
    for t in [1., 2., 3., 4.]:
        np.putmask(newv, np.all((np.abs(v) > t, np.abs(v) <= t+1), axis=0), t)
    np.putmask(newv, np.abs(v) > 5., 10.)
    # -> newv takes values 0, 1, 3, 4, 10
    im.set_var(newv, 'categ', 0) # insert variable in image im

    # ===== Ex4 =====
    pp = pv.Plotter(shape=(2,3))

    pp.subplot(0,0)
    drawImage3D_volume(im, plotter=pp,
        scalar_bar_kwargs={'title':''}, # distinct title in each subplot for correct display!
        text='Ex4: volume (categ. var.)')

    pp.subplot(0,1)
    drawImage3D_surface(im, plotter=pp, categ=False,
        scalar_bar_kwargs={'title':' '}, # distinct title in each subplot for correct display!
        text='Ex4: surface - categ=False')

    cx, cy, cz = im.ox+0.5*im.nx*im.sx, im.oy+0.5*im.ny*im.sy, im.oz+0.5*im.nz*im.sz # center of image
    pp.subplot(0,2)
    drawImage3D_slice(im, plotter=pp, categ=False,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        scalar_bar_kwargs={'title':'  '}, # distinct title in each subplot for correct display!
        text='Ex4: slice - categ=False')

    pp.subplot(1,1)
    drawImage3D_surface(im, plotter=pp, categ=True,
        scalar_bar_kwargs={'title':'   '}, # distinct title in each subplot for correct display!
        text='Ex4: surface - categ=True')

    cx, cy, cz = im.ox+0.5*im.nx*im.sx, im.oy+0.5*im.ny*im.sy, im.oz+0.5*im.nz*im.sz # center of image
    pp.subplot(1,2)
    drawImage3D_slice(im, plotter=pp, categ=True,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        scalar_bar_kwargs={'title':'    '}, # distinct title in each subplot for correct display!
        text='Ex4: slice - categ=True')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # Using some options
    # -------------------
    cols=['purple', 'blue', 'cyan', 'yellow', 'red', 'pink']

    # ===== Ex5 =====
    # Multiple plot
    pp = pv.Plotter(shape=(2,3))

    pp.subplot(0,0)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[True, False, False, False, False, False],
        scalar_bar_kwargs={'title':''}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "0"')

    pp.subplot(0,1)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[False, True, False, False, False, False],
        scalar_bar_kwargs={'title':' '}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "1"')

    pp.subplot(0,2)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[False, False, True, False, False, False],
        scalar_bar_kwargs={'title':'  '}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "2"')

    pp.subplot(1,0)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[False, False, False, True, False, False],
        scalar_bar_kwargs={'title':'   '}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "3"')

    pp.subplot(1,1)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[False, False, False, False, True, False],
        scalar_bar_kwargs={'title':'    '}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "4"')

    pp.subplot(1,2)
    drawImage3D_surface(im, plotter=pp, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols,
        categActive=[False, False, False, False, False, True],
        scalar_bar_kwargs={'title':'     '}, # distinct title in each subplot for correct display!
        text='Ex5: surface - categ=True\n - active categ "10"')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # ===== Ex6 =====
    # activate only some categories
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        text='Ex6: surface - categ=True - active some categ.')

    # ===== Ex7 =====
    # do not show outline
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        show_outline=False, text='Ex7: no outline')

    # ===== Ex8 =====
    # enlarge outline
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        show_outline=True, outline_kwargs={'line_width':5}, text='Ex8: thick outline')

    # ===== Ex9 =====
    # show bounds
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        show_bounds=True, text='Ex9: show bounds')

    # ===== Ex10 =====
    # show bounds with grid
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        show_bounds=True, bounds_kwargs={'grid':True}, text='Ex10: bounds and grid')

    # ===== Ex11 =====
    # customize scalar bar
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        scalar_bar_annotations={0.5:'A', 1.5:'B', 2.5:'C', 3.5:'D', 4.5:'E', 5.5:'high'}, scalar_bar_kwargs={'vertical':True, 'title_font_size':24, 'label_font_size':10},
        text='Ex11: custom scalar bar')

    # ===== Ex12 =====
    # scalar bar: interactive position...
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        scalar_bar_kwargs={'interactive':True}, text='Ex12: interactive scalar bar')

    # ===== Ex13 =====
    # customize title
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        text='Ex13: custom title', text_kwargs={'font_size':12, 'position':'upper_right'})

    # ===== Ex14 =====
    # customize axes
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        axes_kwargs={'x_color':'pink', 'zlabel':'depth'}, text='Ex14: custom axes')

    # ===== Ex15 =====
    # changing background / foreground colors
    drawImage3D_surface(im, categ=True, categVal=[0, 1, 2, 3, 4, 10], categCol=cols, categActive=[True, False, False, True, False, True],
        background_color=(0.9, 0.9, 0.9), foreground_color='k', text='Ex15: background/foreground colors')

    # ===== Ex16 =====
    drawImage3D_surface(im, cmin=2, cmax=4, text='Ex16: surface (categ. var) - categ=False - cmin / cmax')
