#!/usr/bin/python3
#-*- coding: utf-8 -*-

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

# ----------------------------------------------------------------------------
def drawImage3D_surface (
                 im,
                 plotter=None,
                 ix0=0, ix1=None,
                 iy0=0, iy1=None,
                 iz0=0, iz1=None,
                 iv=0,
                 cmap='viridis',
                 cmin=None, cmax=None,
                 custom_scalar_bar_for_equidistant_categories=False,
                 custom_colors=None,
                 opacity=1.0,
                 filtering_value=None,
                 filtering_interval=None,
                 excluded_value=None,
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
                 cpos=None):
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
                        automatically computed if None

    :param custom_scalar_bar_for_equidistant_categories:
                    (bool) indicates if a custom scalar/color bar is drawn
                        (should be used only for categorical variable with
                        equidistant categories)
                        if True: cmin and cmax are automatically computed

    :param custom_colors:
                    (list of colors) for each category,
                        used only if custom_scalar_bar_for_equidistant_categories
                        is set to True, length must be equal to the total number
                        of categories

    :param opacity: (float) between 0.0 and 1.0, opacity used for the plot

    :param filtering_value:
                    (int/float or sequence or None) values to be plotted

    :param filtering_interval:
                    (sequence of length 2, or list of sequence of length 2, or None)
                        interval of values to be plotted

    :param excluded_value:
                    (int/float or sequence or None) values to be
                        excluded from the plot (set to np.nan)

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

    :param cpos:    (None or sequence of three float) position of the camera
                        (unsused if plotter is None)

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print("ERROR: invalid iv index!")
        return

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print("ERROR: invalid cmap string!")
            return

    # Initialization of dictionary (do not used {} as default argument, it is not re-initialized...)
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

    if custom_scalar_bar_for_equidistant_categories:
        all_val = np.unique(zz[~np.isnan(zz)])
        n_all_val = len(all_val)
        s = (all_val[-1] - all_val[0]) / (n_all_val - 1)
        cmin = all_val[0]
        cmax = all_val[-1] + s
        if scalar_bar_annotations == {}:
            for v in all_val:
                scalar_bar_annotations[v+0.5*s]='{:g}'.format(v)
            scalar_bar_kwargs['n_labels'] = 0

        scalar_bar_kwargs['n_colors'] = n_all_val

        if custom_colors is not None:
            if len(custom_colors) != n_all_val:
                print ('ERROR: custom_colors length is not equal to the total number of categories')
                return
            # set cmap (as in geone.customcolor.custom_cmap)
            cols = custom_colors + [custom_colors[-1]] # duplicate last col...

            cseqRGB = []
            for c in cols:
                try:
                    cseqRGB.append(mcolors.ColorConverter().to_rgb(c))
                except:
                    cseqRGB.append(c)

            vseqn = np.linspace(0,1,len(cols))

            # Set dictionary to define the color map
            cdict = {
                'red'  :[(vseqn[i], cseqRGB[i][0], cseqRGB[i][0]) for i in range(len(cols))],
                'green':[(vseqn[i], cseqRGB[i][1], cseqRGB[i][1]) for i in range(len(cols))],
                'blue' :[(vseqn[i], cseqRGB[i][2], cseqRGB[i][2]) for i in range(len(cols))],
                'alpha':[(vseqn[i], 1.0,           1.0)           for i in range(len(cols))]
                }

            cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict, N=256)

    if excluded_value is not None:
        for val in np.array(excluded_value).reshape(-1): # force to be an 1d array
            np.putmask(zz, zz == val, np.nan)

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    # Set pyvista UniformGrid
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    pg = pv.UniformGrid((xdim, ydim, zdim), (im.sx, im.sy, im.sz), (xmin, ymin, zmin))

    pg.cell_arrays[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    if filtering_interval is not None:
        for vv in np.reshape(filtering_interval, (-1,2)):
            pp.add_mesh(pg.threshold(value=vv), cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if filtering_value is not None:
        for v in np.reshape(filtering_value, -1):
            pp.add_mesh(pg.threshold(value=(v,v)), color=cmap((v-cmin)/(cmax-cmin)), show_scalar_bar=False)

    if filtering_value is None and filtering_interval is None:
        pp.add_mesh(pg.threshold(value=(cmin, cmax)), cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        pg.cell_arrays[im.varname[iv]][...] = np.nan # trick: set all value to nan and use nan_opacity = 0 for empty plot but 'saving' the scalar bar...
        # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=True, scalar_bar_args=scalar_bar_kwargs)
        pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
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
                 cmap='viridis',
                 cmin=None, cmax=None,
                 custom_scalar_bar_for_equidistant_categories=False,
                 custom_colors=None,
                 opacity=1.0,
                 nan_color='gray', nan_opacity=1.0,
                 filtering_value=None,
                 filtering_interval=None,
                 excluded_value=None,
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
                 cpos=None):
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

    :param cmap:    colormap (e.g. plt.get_cmap('viridis'), or equivalently,
                        just the string 'viridis' (default))

    :param cmin, cmax:
                    (float) min and max values for the color bar
                        automatically computed if None

    :param custom_scalar_bar_for_equidistant_categories:
                    (bool) indicates if a custom scalar/color bar is drawn
                        (should be used only for categorical variable with
                        equidistant categories)
                        if True: cmin and cmax are automatically computed

    :param custom_colors:
                    (list of colors) for each category,
                        used only if custom_scalar_bar_for_equidistant_categories
                        is set to True, length must be equal to the total number
                        of categories

    :param opacity: (float) between 0.0 and 1.0, opacity used for the plot
    :param nan_color, nan_opacity:
                    color and opacity (float) used for np.nan value

    :param slice_normal_x:
                    (int/float or sequence or None) values of the (real) x
                        coordinate where a slice normal to x-axis is drawn

    :param slice_normal_y:
                    (int/float or sequence or None) values of the (real) y
                        coordinate where a slice normal to y-axis is drawn

    :param slice_normal_z:
                    (int/float or sequence or None) values of the (real) z
                        coordinate where a slice normal to z-axis is drawn

    :param slice_normal_normal:
                    ((sequence of) sequence containing 2 tuple of length 3 or None)
                        slice_normal[i] = ((vx, vy, vz), (px, py, pz))
                        means that a slice normal to the vector (vx, vy, vz) and
                        going through the point (px, py, pz) is drawn

    :param filtering_value:
                    (int/float or sequence or None) values to be plotted

    :param filtering_interval:
                    (sequence of length 2, or list of sequence of length 2, or None)
                        interval of values to be plotted

    :param excluded_value:
                    (int/float or sequence or None) values to be
                        excluded from the plot (set to np.nan)

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

    :param cpos:    (None or sequence of three float) position of the camera
                        (unsused if plotter is None)

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print("ERROR: invalid iv index!")
        return

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print("ERROR: invalid cmap string!")
            return

    # Initialization of dictionary (do not used {} as default argument, it is not re-initialized...)
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

    if custom_scalar_bar_for_equidistant_categories:
        all_val = np.unique(zz[~np.isnan(zz)])
        n_all_val = len(all_val)
        s = (all_val[-1] - all_val[0]) / (n_all_val - 1)
        cmin = all_val[0]
        cmax = all_val[-1] + s
        if scalar_bar_annotations == {}:
            for v in all_val:
                scalar_bar_annotations[v+0.5*s]='{:g}'.format(v)
            scalar_bar_kwargs['n_labels'] = 0

        scalar_bar_kwargs['n_colors'] = n_all_val

        if custom_colors is not None:
            if len(custom_colors) != n_all_val:
                print ('ERROR: custom_colors length is not equal to the total number of categories')
                return
            # set cmap (as in geone.customcolor.custom_cmap)
            cols = custom_colors + [custom_colors[-1]] # duplicate last col...

            cseqRGB = []
            for c in cols:
                try:
                    cseqRGB.append(mcolors.ColorConverter().to_rgb(c))
                except:
                    cseqRGB.append(c)

            vseqn = np.linspace(0,1,len(cols))

            # Set dictionary to define the color map
            cdict = {
                'red'  :[(vseqn[i], cseqRGB[i][0], cseqRGB[i][0]) for i in range(len(cols))],
                'green':[(vseqn[i], cseqRGB[i][1], cseqRGB[i][1]) for i in range(len(cols))],
                'blue' :[(vseqn[i], cseqRGB[i][2], cseqRGB[i][2]) for i in range(len(cols))],
                'alpha':[(vseqn[i], 1.0,           1.0)           for i in range(len(cols))]
                }

            cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict, N=256)

    if filtering_interval is not None or filtering_value is not None:
        bb = np.ones(len(zz)).astype('bool')
        if filtering_interval is not None:
            for vv in np.reshape(filtering_interval, (-1,2)):
                bb = np.all((bb, np.any((np.isnan(zz), zz < vv[0], zz > vv[1]), axis=0)), axis=0)

        if filtering_value is not None:
            for v in np.reshape(filtering_value, -1):
                bb = np.all((bb, zz != v), axis=0)

        zz[bb] = np.nan

    if excluded_value is not None:
        for val in np.array(excluded_value).reshape(-1): # force to be an 1d array
            np.putmask(zz, zz == val, np.nan)

    # Set cmin and cmax if not specified
    if cmin is None:
        cmin = np.nanmin(zz)

    if cmax is None:
        cmax = np.nanmax(zz)

    # Set pyvista UniformGrid
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    pg = pv.UniformGrid((xdim, ydim, zdim), (im.sx, im.sy, im.sz), (xmin, ymin, zmin))

    pg.cell_arrays[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    if slice_normal_x is not None:
        for v in np.array(slice_normal_x).reshape(-1):
            pp.add_mesh(pg.slice(normal=(1,0,0), origin=(v,0,0)), opacity=opacity, nan_color=nan_color, nan_opacity=nan_opacity, cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if slice_normal_y is not None:
        for v in np.array(slice_normal_y).reshape(-1):
            pp.add_mesh(pg.slice(normal=(0,1,0), origin=(0,v,0)), opacity=opacity, nan_color=nan_color, nan_opacity=nan_opacity, cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if slice_normal_z is not None:
        for v in np.array(slice_normal_z).reshape(-1):
            pp.add_mesh(pg.slice(normal=(0,0,1), origin=(0,0,v)), opacity=opacity, nan_color=nan_color, nan_opacity=nan_opacity, cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if slice_normal_custom is not None:
        for nor, ori in np.array(slice_normal_custom).reshape(-1, 2, 3):
            pp.add_mesh(pg.slice(normal=nor, origin=ori), opacity=opacity, nan_color=nan_color, nan_opacity=nan_opacity, cmap=cmap, clim=(cmin, cmax), show_scalar_bar=False)

    if background_color is not None:
        pp.background_color = background_color

    if foreground_color is not None:
        for d in [scalar_bar_kwargs, outline_kwargs, bounds_kwargs, axes_kwargs, text_kwargs]:
            if 'color' not in d.keys():
                d['color'] = foreground_color

    if show_scalar_bar:
        pg.cell_arrays[im.varname[iv]][...] = np.nan # trick: set all value to nan and use nan_opacity = 0 for empty plot but 'saving' the scalar bar...
        # pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=True, scalar_bar_args=scalar_bar_kwargs)
        pp.add_mesh(pg, cmap=cmap, clim=(cmin, cmax), nan_opacity=0., annotations=scalar_bar_annotations, show_scalar_bar=False)
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
def drawImage3D_volume (
                 im,
                 plotter=None,
                 ix0=0, ix1=None,
                 iy0=0, iy1=None,
                 iz0=0, iz1=None,
                 iv=0,
                 cmap='viridis',
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
                 cpos=None):
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

    :param cpos:    (None or sequence of three float) position of the camera
                        (unsused if plotter is None)

    NOTE: 'scalar bar', and 'axes' may be not displayed in multiple-plot, bug ?
    """

    # Check iv
    if iv < 0:
        iv = im.nv + iv

    if iv < 0 or iv >= im.nv:
        print("ERROR: invalid iv index!")
        return

    # Set indices to be plotted
    if ix1 is None:
        ix1 = im.nx

    if iy1 is None:
        iy1 = im.ny

    if iz1 is None:
        iz1 = im.nz

    if ix0 >= ix1 or ix0 < 0 or ix1 > im.nx:
        print("Invalid indices along x)")
        return

    if iy0 >= iy1 or iy0 < 0 or iy1 > im.ny:
        print("Invalid indices along y)")
        return

    if iz0 >= iz1 or iz0 < 0 or iz1 > im.nz:
        print("Invalid indices along z)")
        return

    # Get the color map
    if isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
        except:
            print("ERROR: invalid cmap string!")
            return

    # Initialization of dictionary (do not used {} as default argument, it is not re-initialized...)
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

    # Set pyvista UniformGrid
    xmin = im.ox + ix0 * im.sx
    xmax = im.ox + ix1 * im.sx
    xdim = ix1 - ix0 + 1

    ymin = im.oy + iy0 * im.sy
    ymay = im.oy + iy1 * im.sy
    ydim = iy1 - iy0 + 1

    zmin = im.oz + iz0 * im.sz
    zmaz = im.oz + iz1 * im.sz
    zdim = iz1 - iz0 + 1

    pg = pv.UniformGrid((xdim, ydim, zdim), (im.sx, im.sy, im.sz), (xmin, ymin, zmin))

    pg.cell_arrays[im.varname[iv]] = zz #.flatten()

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    # pp.add_volume(pg.ctp(), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=show_scalar_bar, scalar_bar_args=scalar_bar_kwargs)
    pp.add_volume(pg.ctp(), cmap=cmap, clim=(cmin, cmax), annotations=scalar_bar_annotations, show_scalar_bar=False)
    if show_scalar_bar:
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

if __name__ == "__main__":
    print("Module 'geone.imgplot3d' example:")

    # Example with a 3D gaussian random field

    import geone.covModel as gcm
    import geone.grf as grf

    from geone import img

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

    # Get covariance function and range
    cov_fun = cov_model.func()

    # Set minimal extension according to the size of the grid and the range
    extensionMin = [grf.extension_min(r, n, s) for r, n, s in zip(cov_model.rxyz(), dimension, spacing)]

    # Set seed
    np.random.seed(123)

    # Generate GRF
    v = grf.grf3D(cov_fun, (nx, ny, nz), (dx, dy, dz), (ox, oy, oz), extensionMin=extensionMin)
    im = img.Img(nx, ny, nz, dx, dy, dz, ox, oy, oz, nv=1, val=v)

    # Simple plot
    # ------
    drawImage3D_volume(im, text='im - 1: volume')

    # # Equivalent:
    # pp = pv.Plotter()
    # drawImage3D_volume(im, text='im - 1')
    # pp.show()

    # # For saving screenshot (png)
    # # Note: axes will not be displayed in off screen mode
    # pp = pv.Plotter(off_screen=True)
    # drawImage3D_volume(im, plotter=pp)
    # pp.show(screenshot='test.png')

    # Multiple plot
    # ------
    # Note: scalar bar and axes may be not displayed in all plots (even if show_... option is set to True)
    pp = pv.Plotter(shape=(2,2))

    pp.subplot(0,0)
    drawImage3D_surface(im, plotter=pp, text='im - 2: surface' )

    pp.subplot(0,1)
    drawImage3D_volume(im, plotter=pp, text='im - 2: volume')

    cx, cy, cz = im.ox+0.5*im.nx*im.sx, im.oy+0.5*im.ny*im.sy, im.oz+0.5*im.nz*im.sz # center of image
    pp.subplot(1,0)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text='im - 2: slice')

    pp.subplot(1,1)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_custom=[[(1, 1, 0), (cx, cy, cz)], [(1, -1, 0), (cx, cy, cz)]],
        text='im - 2: slice')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # Categorize image...
    # -------------------
    v = im.val.reshape(-1)
    newv = np.zeros(im.nxyz())
    for t in [1., 2., 3., 4.]:
        np.putmask(newv, np.all((np.abs(v) > t, np.abs(v) <= t+1), axis=0), t)
    np.putmask(newv, np.abs(v) > 5., 5.)
    im.set_var(newv, 'categ', 0)

    pp = pv.Plotter(shape=(2,2))

    pp.subplot(0,0)
    drawImage3D_surface(im, plotter=pp, text='im - 3: surface')

    pp.subplot(0,1)
    drawImage3D_volume(im, plotter=pp, text='im - 3: volume')

    cx, cy, cz = im.ox+0.5*im.nx*im.sx, im.oy+0.5*im.ny*im.sy, im.oz+0.5*im.nz*im.sz # center of image
    pp.subplot(1,0)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text='im - 3: slice')

    pp.subplot(1,1)
    drawImage3D_slice(im, plotter=pp,
        slice_normal_custom=[[(1, 1, 0), (cx, cy, cz)], [(1, -1, 0), (cx, cy, cz)]],
        text='im - 3: slice')

    pp.link_views()
    pp.show(cpos=(1,2,.5))

    # 5. Using some options
    # ------
    cols=['purple', 'blue', 'cyan', 'yellow', 'red', 'pink']

    drawImage3D_surface(im, text='im - 4: surface')
    drawImage3D_surface(im, custom_scalar_bar_for_equidistant_categories=True, text='im - 5: custom scalar bar')
    drawImage3D_surface(im, custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, text='im - 6: custom scalar bar (2)')
    drawImage3D_surface(im, filtering_value=[1, 5], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, text='im - 7: filtering')

    # # filtering does not change cmin, cmax, compare:
    # drawImage3D_surface(im, cmin=2, cmax=4, text='im - 7b')
    # drawImage3D_surface(im, filtering_interval=[2, 4], text='im - 7c')

    # do not show outline
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, show_outline=False, text='im - 8: no outline')

    # enlarge outline
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, show_outline=True, outline_kwargs={'line_width':5}, text='im - 9: thick outline')

    # show bounds
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, show_bounds=True, text='im - 10: bounds')

    # show bounds with grid
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, show_bounds=True, bounds_kwargs={'grid':True}, text='im - 11: bounds and grid')

    # customize scalar bar
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, scalar_bar_kwargs={'vertical':True,'title_font_size':24, 'label_font_size':10}, text='im - 12: custom display of scalar bar')

    # scalar bar: interactive position...
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, scalar_bar_kwargs={'interactive':True}, text='im - 13: interactive display of scalar bar')

    # customize title
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, text='im - 14: custom title', text_kwargs={'font_size':12, 'position':'upper_right'})

    # customize axes
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, axes_kwargs={'x_color':'pink', 'zlabel':'depth'}, text='im - 15: custom axes')

    # changing background / foreground colors
    drawImage3D_surface(im, filtering_interval=[2, 4], custom_scalar_bar_for_equidistant_categories=True, custom_colors=cols, background_color=(0.9, 0.9, 0.9), foreground_color='k', text='im - 16: background/foreground colors')

    # # (less options for drawImage3D_volume)
    # drawImage3D_volume(im, text='im - 17')
    # drawImage3D_volume(im, cmin=2, cmax=4, text='im - 18')
    # drawImage3D_volume(im, cmin=2, cmax=4, set_out_values_to_nan=False, text='im - 19')
