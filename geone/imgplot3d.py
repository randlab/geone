#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'imgplot3d.py'
author:         Julien Straubhaar
date:           feb-2020

Definition of functions for plotting images (geone.Img class) in 3d based on
pyvista.
"""

# from geone import customcolors as ccol

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import pyvista as pv

from geone import customcolors as ccol

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
        scalar_bar_annotations_max=30,
        scalar_bar_kwargs=None,
        outline_kwargs=None,
        bounds_kwargs=None,
        axes_kwargs=None,
        text_kwargs=None,
        background_color=None,
        foreground_color=None,
        cpos=None,
        **kwargs):
    """
    Draws a 3D image as surface(s) (using pyvista):

    :param im:      (img.Img class) image (3D)

    :param plotter: (pyvista plotter)
                        if given: add element to the plotter, a further call
                            to plotter.show() will be required to show the plot
                        if None (default): a plotter is created and the plot
                            is shown

    :param ix0, ix1:(int or None) indices for x direction ix0 < ix1
                        indices ix0:ix1 will be considered for plotting
                        if ix1 is None (default): ix1 will be set to im.nx

    :param iy0, iy1:(int or None) indices for y direction iy0 < iy1
                        indices iy0:iy1 will be considered for plotting
                        if iy1 is None (default): iy1 will be set to im.ny

    :param iz0, iz1:(int or None) indices for z direction iz0 < iz1
                        indices iz0:iz1 will be considered for plotting
                        if iz1 is None (default): iz1 will be set to im.nz


    :param iv:      (int) index of the variable to be drawn

    :param cmap:    colormap (e.g. plt.get_cmap('viridis'), or equivalently,
                        just the string 'viridis' (default))

    :param cmin, cmax:
                    (float) min and max values for the color bar
                        -- used only if categ is False --
                        automatically computed if None

    :param alpha:   (float or None) values of alpha channel for transparency
                        (if None, value 1.0 is used (no transparency))

    :param excludedVal: (int/float or sequence or None) values to be
                            excluded from the plot.
                            Note: not used if categ is True and categVal is
                            not None

    :param categ:       (bool) indicates if the variable of the image to plot
                            has to be treated as categorical (True) or as
                            continuous (False)

    :param ncateg_max:  (int) maximal number of categories
                            -- used only if categ is True --
                            if more category values and categVal is None,
                            nothing is plotted (categ should set to False)

    :param categVal:    (int/float or sequence or None)
                            -- used only if categ is True --
                            explicit list of the category values to be
                            considered (if None, the list of all unique values
                            are automatically computed)

    :param categCol:    (sequence or None)
                            -- used only if categ is True --
                            colors (given by string or rgb-tuple) used for the
                            category values that will be displayed:
                                If categVal is not None: categCol must have
                                    the same length as categVal,
                                else: first entries of categCol are used if its
                                    length is greater or equal to the number
                                    of displayed category values, otherwise:
                                    the entries of categCol are used cyclically
                                    if categColCycle is True and otherwise,
                                    colors taken from the colormap cmap are used

    :param categColCycle:
                        (bool)
                            -- used only if categ is True --
                            indicates if the entries of categCol can be used
                            cyclically or not (when the number of displayed
                            category values exceeds the length of categCol)

    :param categActive:
                    (sequence of bools or None)
                        -- used only if categ is True --
                        sequence of same length as categVal:
                        - categActive[i] is True: categVal[i] is displayed
                        - categActive[i] is False: categVal[i] is not displayed
                        if None, all category values (in categVal) is displayed

    :param use_clip_plane:
                    (bool) set True to use 'pyvista.add_mesh_clip_plane'
                        (allowing interactive clipping) instead of
                        'pyvista.add_mesh' when plotting values of the image;
                        warning: one clip plane per value (resp. interval)
                        specified in filtering_value (resp. filtering_intervals)
                        is generated

    :param show_scalar_bar:
                    (bool) indicates if scalar bar (color bar) is drawn

    :param show_outline:
                    (bool) indicates if outline (around the image) is drawn

    :param show_bounds:
                    (bool) indicates if bounds are drawn (box with graduation)

    :param show_axes:
                    (bool) indicates if axes are drawn

    :param text:    (string or None) text to be written on the figure (title)

    :param scalar_bar_annotations:
                    (dict) annotation on the scalar bar (color bar)
                        (used if show_scalar_bar is True)

    :param scalar_bar_annotations_max:
                    (int) maximal number of annotations on the scalar bar
                        when categ is True and scalar_bar_annotations is None

    :param scalar_bar_kwargs:
                    (dict) kwargs passed to function 'plotter.add_scalar_bar'
                        (useful for customization,
                        used if show_scalar_bar is True)
                        Note: in subplots (multi-sub-window), key 'title' should
                        be distinct for each subplot

    :param outline_kwargs:
                    (dict) kwargs passed to function 'plotter.add_mesh'
                        (useful for customization,
                        used if show_outline is True)

    :param bounds_kwargs:
                    (dict) kwargs passed to function 'plotter.show_bounds'
                        (useful for customization,
                        used if show_bounds is True)

    :param axes_kwargs:
                    (dict) kwargs passed to function 'plotter.add_axes'
                        (useful for customization,
                        used if show_axes is True)

    :param text_kwargs:
                    (dict) kwargs passed to function 'plotter.add_text'
                        (useful for customization,
                        used if text is not None)

    :param background_color:
                    background color

    :param foreground_color:
                    foreground color

    :param cpos:
            (list of three 3-tuples, or None for default) camera position
                (unsused if plotter is None)
                cpos = [camera_location, focus_point, viewup_vector], with
                camera_location: (tuple of length 3) camera location ("eye")
                focus_point    : (tuple of length 3) focus point
                viewup_vector  : (tuple of length 3) viewup vector (vector
                    attached to the "head" and pointed to the "sky"),
                    in principle: (focus_point - camera_location) is orthogonal
                    to viewup_vector

    :param kwargs:
        additional keyword arguments passed to plotter.add_mesh[_clip_plane] when
        plotting the variable, such as
            - opacity:  (float or string) opacity for colors
                            default: 'linear', (set 'linear_r' to invert opacity)
            - show_edges:
                        (bool) indicates if edges of the grid are drawn
            - edge_color:
                        (string or 3 item list) color for edges (used if
                            show_edges is True)
            - line_width:
                        (float) line width for edges (used if show_edges is True)
            - etc.

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    fname = 'drawImage3D_surface'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print(f'ERROR ({fname}): invalid iv index!')
        return None

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return None

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return None

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return None

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print(f'ERROR ({fname}): invalid cmap string!')
            return None

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
        if categCol is not None\
                and type(categCol) is not list\
                and type(categCol) is not tuple:
            print(f"ERROR ({fname}): 'categCol' must be a list or a tuple (if not None)!")
            return None

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                print(f"ERROR ({fname}): 'categVal' contains duplicated entries!")
                return None

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                print(f"ERROR ({fname}): length of 'categVal' and 'categCol' differs!")
                return None

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                print(f'ERROR ({fname}): too many categories, set categ=False')
                return None

        if not len(dval): # len(dval) == 0
            print(f'ERROR ({fname}): no value to be drawn!')
            return None

        if categActive is not None:
            if len(categActive) != len(dval):
                print(f"ERROR ({fname}): length of 'categActive' not valid (should be the same as length of categVal)")
                return None
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
                print("Warning: 'categCol' is used cyclically (too few entries)")
                colorList = [categCol[i%len(categCol)] for i in range(len(dval))]

            else:
                print("Warning: 'categCol' not used (too few entries)")

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(x) for x in np.arange(len(dval)) * 1.0/(len(dval)-1)]

        # Set the colormap: 'cmap'
        # - Trick: duplicate last color (even if len(colorList)> 1)!
        #          otherwise the first color appears twice
        colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5]='{:.3g}'.format(v)

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
        rendering='volume',
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
        **kwargs):
    """
    Draws a 3D image as slice(s) (using pyvista):

    :param im:      (img.Img class) image (3D)

    :param plotter: (pyvista plotter)
                        if given: add element to the plotter, a further call
                            to plotter.show() will be required to show the plot
                        if None (default): a plotter is created and the plot
                            is shown

    :param ix0, ix1:(int or None) indices for x direction ix0 < ix1
                        indices ix0:ix1 will be considered for plotting
                        if ix1 is None (default): ix1 will be set to im.nx

    :param iy0, iy1:(int or None) indices for y direction iy0 < iy1
                        indices iy0:iy1 will be considered for plotting
                        if iy1 is None (default): iy1 will be set to im.ny

    :param iz0, iz1:(int or None) indices for z direction iz0 < iz1
                        indices iz0:iz1 will be considered for plotting
                        if iz1 is None (default): iz1 will be set to im.nz

    :param iv:      (int) index of the variable to be drawn

    :param slice_normal_x:
                    (int/float or sequence or None) values of the (real) x
                        coordinate where a slice normal to x-axis is drawn

    :param slice_normal_y:
                    (int/float or sequence or None) values of the (real) y
                        coordinate where a slice normal to y-axis is drawn

    :param slice_normal_z:
                    (int/float or sequence or None) values of the (real) z
                        coordinate where a slice normal to z-axis is drawn

    :param slice_normal_custom:
                    ((sequence of) sequence containing 2 tuple of length 3 or
                        None) slice_normal[i] = ((vx, vy, vz), (px, py, pz))
                        means that a slice normal to the vector (vx, vy, vz) and
                        going through the point (px, py, pz) is drawn

    :param cmap:    colormap (e.g. plt.get_cmap('viridis'), or equivalently,
                        just the string 'viridis' (default))

    :param cmin, cmax:
                    (float) min and max values for the color bar
                        -- used only if categ is False --
                        automatically computed if None

    :param alpha:   (float or None) values of alpha channel for transparency
                        (if None, value 1.0 is used (no transparency))

    :param excludedVal: (int/float or sequence or None) values to be
                            excluded from the plot.
                            Note: not used if categ is True and categVal is
                            not None

    :param categ:       (bool) indicates if the variable of the image to plot
                            has to be treated as categorical (True) or as
                            continuous (False)

    :param ncateg_max:  (int) maximal number of categories
                            -- used only if categ is True --
                            if more category values and categVal is None,
                            nothing is plotted (categ should set to False)

    :param categVal:    (int/float or sequence or None)
                            -- used only if categ is True --
                            explicit list of the category values to be
                            considered (if None, the list of all unique values
                            are automatically computed)

    :param categCol:    (sequence or None)
                            -- used only if categ is True --
                            colors (given by string or rgb-tuple) used for the
                            category values that will be displayed:
                                If categVal is not None: categCol must have
                                    the same length as categVal,
                                else: first entries of categCol are used if its
                                    length is greater or equal to the number
                                    of displayed category values, otherwise:
                                    the entries of categCol are used cyclically
                                    if categColCycle is True and otherwise,
                                    colors taken from the colormap cmap are used

    :param categColCycle:
                        (bool)
                            -- used only if categ is True --
                            indicates if the entries of categCol can be used
                            cyclically or not (when the number of displayed
                            category values exceeds the length of categCol)

    :param categActive:
                    (sequence of bools or None)
                        -- used only if categ is True --
                        sequence of same length as categVal:
                        - categActive[i] is True: categVal[i] is displayed
                        - categActive[i] is False: categVal[i] is not displayed
                        if None, all category values (in categVal) is displayed

    :param show_scalar_bar:
                    (bool) indicates if scalar bar (color bar) is drawn

    :param show_outline:
                    (bool) indicates if outline (around the image) is drawn

    :param show_bounds:
                    (bool) indicates if bounds are drawn (box with graduation)

    :param show_axes:
                    (bool) indicates if axes are drawn

    :param text:    (string or None) text to be written on the figure (title)

    :param scalar_bar_annotations:
                    (dict) annotation on the scalar bar (color bar)
                        (used if show_scalar_bar is True)

    :param scalar_bar_kwargs:
                    (dict) kwargs passed to function 'plotter.add_scalar_bar'
                        (useful for customization,
                        used if show_scalar_bar is True)
                        Note: in subplots (multi-sub-window), key 'title' should
                        be distinct for each subplot

    :param outline_kwargs:
                    (dict) kwargs passed to function 'plotter.add_mesh'
                        (useful for customization,
                        used if show_outline is True)

    :param bounds_kwargs:
                    (dict) kwargs passed to function 'plotter.show_bounds'
                        (useful for customization,
                        used if show_bounds is True)

    :param axes_kwargs:
                    (dict) kwargs passed to function 'plotter.add_axes'
                        (useful for customization,
                        used if show_axes is True)

    :param text_kwargs:
                    (dict) kwargs passed to function 'plotter.add_text'
                        (useful for customization,
                        used if text is not None)

    :param background_color:
                    background color

    :param foreground_color:
                    foreground color

    :param cpos:
            (list of three 3-tuples, or None for default) camera position
                (unsused if plotter is None)
                cpos = [camera_location, focus_point, viewup_vector], with
                camera_location: (tuple of length 3) camera location ("eye")
                focus_point    : (tuple of length 3) focus point
                viewup_vector  : (tuple of length 3) viewup vector (vector
                    attached to the "head" and pointed to the "sky"),
                    in principle: (focus_point - camera_location) is orthogonal
                    to viewup_vector

    :param kwargs:
        additional keyword arguments passed to plotter.add_mesh when
        plotting the variable, such as
            - opacity:  (float or string) opacity for colors
                            default: 'linear', (set 'linear_r' to invert opacity)
            - nan_color:
                        color for np.nan value
            - nan_opacity:
                        (float) opacity used for np.nan value
            - show_edges:
                        (bool) indicates if edges of the grid are drawn
            - edge_color:
                        (string or 3 item list) color for edges (used if
                            show_edges is True)
            - line_width:
                        (float) line width for edges (used if show_edges is True)
            - etc.

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    fname = 'drawImage3D_slice'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print(f'ERROR ({fname}): invalid iv index!')
        return None

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return None

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return None

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return None

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print(f'ERROR ({fname}): invalid cmap string!')
            return None

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
        if categCol is not None\
                and type(categCol) is not list\
                and type(categCol) is not tuple:
            print(f"ERROR ({fname}): 'categCol' must be a list or a tuple (if not None)!")
            return None

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                print(f"ERROR ({fname}): 'categVal' contains duplicated entries!")
                return None

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                print(f"ERROR ({fname}): length of 'categVal' and 'categCol' differs!")
                return None

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                print(f'ERROR ({fname}): too many categories, set categ=False')
                return None

        if not len(dval): # len(dval) == 0
            print(f'ERROR ({fname}): no value to be drawn!')
            return None

        if categActive is not None:
            if len(categActive) != len(dval):
                print(f"ERROR ({fname}): length of 'categActive' not valid (should be the same as length of categVal)")
                return None
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
                print("Warning: 'categCol' is used cyclically (too few entries)")
                colorList = [categCol[i%len(categCol)] for i in range(len(dval))]

            else:
                print("Warning: 'categCol' not used (too few entries)")

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(x) for x in np.arange(len(dval)) * 1.0/(len(dval)-1)]

        # Set the colormap: 'cmap'
        # - Trick: duplicate last color (even if len(colorList)> 1)!
        #          otherwise the first color appears twice
        colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5]='{:.3g}'.format(v)

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
        rendering='volume',
        ix0=0, ix1=None,
        iy0=0, iy1=None,
        iz0=0, iz1=None,
        # iv=0,
        # slice_normal_x=None,
        # slice_normal_y=None,
        # slice_normal_z=None,
        # slice_normal_custom=None,
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
        **kwargs):
    """
    Draws an empty grid from a 3D image, see parameters in function
    drawImage3D_slice. Tricks are done below.
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
        print("Invalid indices along x)")
        return None

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return None

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return None

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print(f'ERROR ({fname}): invalid cmap string!')
            return None

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
        if categCol is not None\
                and type(categCol) is not list\
                and type(categCol) is not tuple:
            print(f"ERROR ({fname}): 'categCol' must be a list or a tuple (if not None)!")
            return None

        # Get array 'dval' of displayed values (at least for color bar)
        if categVal is not None:
            dval = np.array(categVal).reshape(-1) # force to be an 1d array

            if len(np.unique(dval)) != len(dval):
                print(f"ERROR ({fname}): 'categVal' contains duplicated entries!")
                return None

            # Check 'categCol' (if not None)
            if categCol is not None and len(categCol) != len(dval):
                print(f"ERROR ({fname}): length of 'categVal' and 'categCol' differs!")
                return None

        else:
            # Possibly exclude values from zz
            if excludedVal is not None:
                for val in np.array(excludedVal).reshape(-1):
                    np.putmask(zz, zz == val, np.nan)

            # Get the unique values in zz
            dval = np.array([v for v in np.unique(zz).reshape(-1) if ~np.isnan(v)])
            if len(dval) > ncateg_max:
                print(f'ERROR ({fname}): too many categories, set categ=False')
                return None

        if not len(dval): # len(dval) == 0
            print(f'ERROR ({fname}): no value to be drawn!')
            return None

        if categActive is not None:
            if len(categActive) != len(dval):
                print(f"ERROR ({fname}): length of 'categActive' not valid (should be the same as length of categVal)")
                return None
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
                print("Warning: 'categCol' is used cyclically (too few entries)")
                colorList = [categCol[i%len(categCol)] for i in range(len(dval))]

            else:
                print("Warning: 'categCol' not used (too few entries)")

        if colorList is None:
            # Use colors from cmap
            colorList = [cmap(x) for x in np.arange(len(dval)) * 1.0/(len(dval)-1)]

        # Set the colormap: 'cmap'
        # - Trick: duplicate last color (even if len(colorList)> 1)!
        #          otherwise the first color appears twice
        colorList.append(colorList[-1])
        cmap = ccol.custom_cmap(colorList, ncol=len(colorList), alpha=alpha)

        # Set the min and max of the colorbar
        cmin, cmax = 0, len(dval) # works, but scalar bar annotations may be shifted of +0.5, see below
        # cmin, cmax = -0.5, len(dval) - 0.5 # does not work

        # Set scalar bar annotations if not given
        if scalar_bar_annotations == {}:
            if len(dval) <= scalar_bar_annotations_max: # avoid too many annotations (very slow and useless)
                for i, v in enumerate(dval):
                    scalar_bar_annotations[i+0.5]='{:.3g}'.format(v)

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
        **kwargs):
    """
    Draws a 3D image as volume (using pyvista):

    :param im:      (img.Img class) image (3D)

    :param plotter: (pyvista plotter)
                        if given: add element to the plotter, a further call
                            to plotter.show() will be required to show the plot
                        if None (default): a plotter is created and the plot
                            is shown

    :param ix0, ix1:(int or None) indices for x direction ix0 < ix1
                        indices ix0:ix1 will be considered for plotting
                        if ix1 is None (default): ix1 will be set to im.nx

    :param iy0, iy1:(int or None) indices for y direction iy0 < iy1
                        indices iy0:iy1 will be considered for plotting
                        if iy1 is None (default): iy1 will be set to im.ny

    :param iz0, iz1:(int or None) indices for z direction iz0 < iz1
                        indices iz0:iz1 will be considered for plotting
                        if iz1 is None (default): iz1 will be set to im.nz


    :param iv:      (int) index of the variable to be drawn

    :param cmap:    colormap (e.g. plt.get_cmap('viridis'), or equivalently,
                        just the string 'viridis' (default))

    :param cmin, cmax:
                    (float) min and max values for the color bar
                        automatically computed if None

    :param set_out_values_to_nan:
                    (bool) indicates if values out of the range [cmin, cmax]
                        is set to np.nan before plotting

    :param show_scalar_bar:
                    (bool) indicates if scalar bar (color bar) is drawn

    :param show_outline:
                    (bool) indicates if outline (around the image) is drawn

    :param show_bounds:
                    (bool) indicates if bounds are drawn (box with graduation)

    :param show_axes:
                    (bool) indicates if axes are drawn

    :param text:    (string or None) text to be written on the figure (title)

    :param scalar_bar_annotations:
                    (dict) annotation on the scalar bar (color bar)
                        (used if show_scalar_bar is True)

    :param scalar_bar_kwargs:
                    (dict) kwargs passed to function 'plotter.add_scalar_bar'
                        (useful for customization,
                        used if show_scalar_bar is True)
                        Note: in subplots (multi-sub-window), key 'title' should
                        be distinct for each subplot

    :param outline_kwargs:
                    (dict) kwargs passed to function 'plotter.add_mesh'
                        (useful for customization,
                        used if show_outline is True)

    :param bounds_kwargs:
                    (dict) kwargs passed to function 'plotter.show_bounds'
                        (useful for customization,
                        used if show_bounds is True)

    :param axes_kwargs:
                    (dict) kwargs passed to function 'plotter.add_axes'
                        (useful for customization,
                        used if show_axes is True)

    :param text_kwargs:
                    (dict) kwargs passed to function 'plotter.add_text'
                        (useful for customization,
                        used if text is not None)

    :param background_color:
                    background color

    :param foreground_color:
                    foreground color

    :param cpos:
            (list of three 3-tuples, or None for default) camera position
                (unsused if plotter is None)
                cpos = [camera_location, focus_point, viewup_vector], with
                camera_location: (tuple of length 3) camera location ("eye")
                focus_point    : (tuple of length 3) focus point
                viewup_vector  : (tuple of length 3) viewup vector (vector
                    attached to the "head" and pointed to the "sky"),
                    in principle: (focus_point - camera_location) is orthogonal
                    to viewup_vector

    :param kwargs:
        additional keyword arguments passed to plotter.add_volume
        such as
            - opacity: (float or string) opacity for colors (see doc of
                            pyvista.add_volume), default: 'linear',
                            (set 'linear_r' to invert opacity)
            - etc.

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    fname = 'drawImage3D_volume'

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print(f'ERROR ({fname}): invalid iv index!')
        return None

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return None

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return None

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return None

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print(f'ERROR ({fname}): invalid cmap string!')
            return None

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
