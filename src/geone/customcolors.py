#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'customcolors.py'
# author:         Julien Straubhaar
# date:           jan-2018
# -------------------------------------------------------------------------

"""
Module for definition of custom colors and colormaps.

**Defined objects in this module**

- "default" color for "bad" value (nan)
    - `cbad_def = (.9, .9, .9, 0.5)` : default color for "bad" value (nan)

- "default" color map
    - `cmap_def`

- some colors from "libreoffice" (merci Christoph!)
    - `col_chart01 = [x/255. for x in (  0,  69, 134)]` : dark blue
    - `col_chart02 = [x/255. for x in (255,  66,  14)]` : orange
    - `col_chart03 = [x/255. for x in (255, 211,  32)]` : yellow
    - `col_chart04 = [x/255. for x in ( 87, 157,  28)]` : green
    - `col_chart05 = [x/255. for x in (126,   0,  33)]` : dark red
    - `col_chart06 = [x/255. for x in (131, 202, 255)]` : light blue
    - `col_chart07 = [x/255. for x in ( 49,  64,   4)]` : dark green
    - `col_chart08 = [x/255. for x in (174, 207,   0)]` : light green
    - `col_chart09 = [x/255. for x in ( 75,  31, 111)]` : purple
    - `col_chart10 = [x/255. for x in (255, 149,  14)]` : dark yellow
    - `col_chart11 = [x/255. for x in (197,   0,  11)]` : red
    - `col_chart12 = [x/255. for x in (  0, 132, 209)]` : blue

- ... with other names
    - `col_chart_purple     = col_chart09`
    - `col_chart_darkblue   = col_chart01`
    - `col_chart_blue       = col_chart12`
    - `col_chart_lightblue  = col_chart06`
    - `col_chart_green      = col_chart04`
    - `col_chart_darkgreen  = col_chart07`
    - `col_chart_lightgreen = col_chart08`
    - `col_chart_yellow     = col_chart03`
    - `col_chart_darkyellow = col_chart10`
    - `col_chart_orange     = col_chart02`
    - `col_chart_red        = col_chart11`
    - `col_chart_darkred    = col_chart05`

- ... in list
    - `col_chart_list = [col_chart01, col_chart02, col_chart03, col_chart04, \
                        col_chart05, col_chart06, col_chart07, col_chart08, \
                        col_chart09, col_chart10, col_chart11, col_chart12]`
    - `col_chart_list_s = [col_chart_list[i] for i in (8, 0, 11, 5, 3, 6, 7, 2, 9, 1, 10, 4)]`

- pre-defined colormaps
    - `cmap1`, `cmap2`, `cmapW2B` (white to black), `cmapB2W` (black to white)
"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1

# ============================================================================
class CustomcolorsError(Exception):
    """
    Custom exception related to `customcolors` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def add_colorbar(im, aspect=20.0, pad_fraction=1.0, **kwargs):
    """
    Adds a vertical color bar to an image plot.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        image (returned by `matplotlib.pyplot.imshow`)

    aspect : float, default: 20.0
        aspect ratio height over width of the color bar

    pad_fraction : float, default: 1.0
        padding fraction with respect to the color bar width; if w is the width
        of the color bar, then the space between the image and the color bar will
        be of with `pad_fraction` * w

    kwargs : dict
        keyword arguments passed to `im.axes.figure.colorbar`

    Notes
    -----
    Code from https://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/plotting/matplotlib-colorbar.ipynb
    """
    # fname = 'add_colorbar'

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def custom_cmap(
        cseq,
        vseq=None,
        ncol=256,
        cunder=None,
        cover=None,
        cbad=None,
        alpha=None,
        cmap_name='custom_cmap',
        logger=None):
    """
    Defines a custom color map given colors at transition values.

    Parameters
    ----------
    cseq : 1D array-like
        sequence of colors, each color is given by a
        3-tuple (RGB code), 4-tuple (RGBA code) or string

    vseq : 1D array-like of floats, optional
        sequence of increasing values of same length as `cseq`, values that
        correspond to the colors given in `cseq` in the output color map;
        by default (`None`): equally spaced values (from 0 to 1) are used

    ncol : int, default: 256
        number of colors for the color map

    cunder : object defining a color (e.g. 3-tuple, str), optional
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        "under" values

    cover : object defining a color (e.g. 3-tuple, str), optional
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        "over" values

    cbad : object defining a color (e.g. 3-tuple, str), optional
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        "bad" values

    alpha : 1D array-like of floats, or float, optional
        values of the "alpha" channel (for transparency), for each color in
        `cseq` (if a `alpha` is a float, the same value is used for every color)

    cmap_name : str, default: 'custom_cmap'
        name of the output color map

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    colormap : `matplotlib.colors.LinearSegmentedColormap`
        color map

    Examples
    --------
        >>> # # Import if needed
        >>> # import numpy as np
        >>> # import matplotlib.pyplot as plt
        >>> #
        >>> # Set min and max value to be displayed
        >>> vmin, vmax = -1.0, 3.0
        >>> #
        >>> # Create a custom colormap
        >>> #   - from blue to white for values from vmin to 0
        >>> #   - from white to red for values from 0 to vmax
        >>> #   - with specified colors for "under", "over" and "bad" values
        >>> my_cmap = custom_cmap(
        >>>     ('blue', 'white', 'red'), vseq=(vmin,0,vmax),
        >>>     cunder='cyan', cover='violet', cbad='gray', alpha=1)
        >>> #
        >>> # Array of values to display
        >>> zz = np.linspace(-2, 4, 500).reshape(10, 50)
        >>> zz[:, 20:25] = np.nan
        >>> #
        >>> # Show the array, specifiying min, max and the color map
        >>> im = plt.imshow(zz, vmin=vmin, vmax=vmax, cmap=my_cmap)
        >>> add_colorbar(im, aspect=5., extend='both') # add color bar
        >>> plt.show()
    """
    fname = 'custom_cmap'

    # Set alpha sequence
    if alpha is None:
        alpha = 1.0
    aseq = np.asarray(alpha, dtype=float) # numpy.ndarray (possibly 0-dimensional)
    if aseq.size == 1:
        aseq = aseq.flat[0] * np.ones(len(cseq))
    elif aseq.size != len(cseq):
        err_msg = f'{fname}: length of `alpha` not compatible with `cseq`'
        if logger: logger.error(err_msg)
        raise CustomcolorsError(err_msg)

    # Set vseqn: sequence of values rescaled in [0,1]
    if vseq is not None:
        if len(vseq) != len(cseq):
            err_msg = f'{fname}: length of `vseq` and length of `cseq` differ'
            if logger: logger.error(err_msg)
            raise CustomcolorsError(err_msg)

        if sum(np.diff(vseq) <= 0.0 ):
            err_msg = f'{fname}: `vseq` is not an increasing sequence'
            if logger: logger.error(err_msg)
            raise CustomcolorsError(err_msg)

        # Linearly rescale vseq on [0,1]
        vseqn = (np.array(vseq,dtype=float) - vseq[0]) / (vseq[-1] - vseq[0])

    else:
        vseqn = np.linspace(0,1,len(cseq))

    # Set cseqRGB: sequence of colors as RGB-tuples
    cseqRGB = []
    for c in cseq:
        try:
            cseqRGB.append(mcolors.ColorConverter().to_rgb(c))
        except:
            cseqRGB.append(c)

    # Set dictionary to define the color map
    cdict = {
        'red'  :[(vseqn[i], cseqRGB[i][0], cseqRGB[i][0]) for i in range(len(cseq))],
        'green':[(vseqn[i], cseqRGB[i][1], cseqRGB[i][1]) for i in range(len(cseq))],
        'blue' :[(vseqn[i], cseqRGB[i][2], cseqRGB[i][2]) for i in range(len(cseq))],
        'alpha':[(vseqn[i], aseq[i],       aseq[i])       for i in range(len(cseq))]
        }

    cmap = mcolors.LinearSegmentedColormap(cmap_name, cdict, N=ncol)

    if cunder is not None:
        try:
            cmap.set_under(mcolors.ColorConverter().to_rgba(cunder))
        except:
            cmap.set_under(cunder)

    if cover is not None:
        try:
            cmap.set_over(mcolors.ColorConverter().to_rgba(cover))
        except:
            cmap.set_over(cover)

    if cbad is not None:
        try:
            cmap.set_bad(mcolors.ColorConverter().to_rgba(cbad))
        except:
            cmap.set_bad(cbad)

    return cmap
# ----------------------------------------------------------------------------

# Default colormap for function in geone
# ======================================
cmap_def = 'viridis'

# Some colors and colormaps
# =========================

# Chart color from libreoffice (merci Christoph)
col_chart01 = [x/255. for x in (  0,  69, 134)]   # dark blue
col_chart02 = [x/255. for x in (255,  66,  14)]   # orange
col_chart03 = [x/255. for x in (255, 211,  32)]   # yellow
col_chart04 = [x/255. for x in ( 87, 157,  28)]   # green
col_chart05 = [x/255. for x in (126,   0,  33)]   # dark red
col_chart06 = [x/255. for x in (131, 202, 255)]   # light blue
col_chart07 = [x/255. for x in ( 49,  64,   4)]   # dark green
col_chart08 = [x/255. for x in (174, 207,   0)]   # light green
col_chart09 = [x/255. for x in ( 75,  31, 111)]   # purple
col_chart10 = [x/255. for x in (255, 149,  14)]   # dark yellow
col_chart11 = [x/255. for x in (197,   0,  11)]   # red
col_chart12 = [x/255. for x in (  0, 132, 209)]   # blue

# ... other names
col_chart_purple     = col_chart09
col_chart_darkblue   = col_chart01
col_chart_blue       = col_chart12
col_chart_lightblue  = col_chart06
col_chart_green      = col_chart04
col_chart_darkgreen  = col_chart07
col_chart_lightgreen = col_chart08
col_chart_yellow     = col_chart03
col_chart_darkyellow = col_chart10
col_chart_orange     = col_chart02
col_chart_red        = col_chart11
col_chart_darkred    = col_chart05

# ... list
col_chart_list = [col_chart01, col_chart02, col_chart03, col_chart04,
                  col_chart05, col_chart06, col_chart07, col_chart08,
                  col_chart09, col_chart10, col_chart11, col_chart12]

# ... list reordered
col_chart_list_s = [col_chart_list[i] for i in (8, 0, 11, 5, 3, 6, 7, 2, 9, 1, 10, 4)]

# Default color for bad value (nan)
cbad_def = (.9, .9, .9, 0.5)

# Pre-defined colormaps
# =====================

# colormap cmap1 / cmap1_alpha (increasing alpha values)
# --------------
cbad1   = (.9, .9, .9, 0.5)
# cunder1 = [x/255. for x in (160, 40, 160)] + [0.5] # +[0.5] ... for appending alpha channel
# cover1  = [x/255. for x in (250,  80, 120)] + [0.5] # +[0.5] ... for appending alpha channel
cunder1 = [x/255. for x in (200, 10, 250)] + [0.5] # +[0.5] ... for appending alpha channel
cover1  = [x/255. for x in (250,  10, 10)] + [0.5] # +[0.5] ... for appending alpha channel
cmaplist1 = ([x/255. for x in (160,  40, 240)],
             [x/255. for x in (  0, 240, 240)],
             [x/255. for x in (240, 240,   0)],
             [x/255. for x in (180,  10,  10)])
cmap1 = custom_cmap(cmaplist1, cunder=cunder1, cover=cover1, cbad=cbad1, alpha=1.0)
cmap1_alpha = custom_cmap(cmaplist1, alpha=np.linspace(0, 1, len(cmaplist1)), cunder=cunder1, cover=cover1, cbad=cbad1)

# colormap cmap2 / cmap2_alpha (increasing alpha values)
# --------------
cmaplist2 = ['purple', 'blue', 'cyan', 'yellow', 'red', 'black']
cbad2   = cbad_def
cunder2 = cbad_def
cover2  = cbad_def
cmap2 = custom_cmap(cmaplist2, cunder=cunder2, cover=cover2, cbad=cbad2, alpha=1.0)
cmap2_alpha = custom_cmap(cmaplist2, alpha=np.linspace(0, 2, len(cmaplist2)), cunder=cunder2, cover=cover2, cbad=cbad2)

# colormap cmapW2B / cmapB2W (white to black / black to white)
# --------------
cmapW2B = custom_cmap(['white', 'black'], cunder=(0.0, 0.0, 1.0, 0.5), cover=(1.0, 0.0, 0.0, 0.5), cbad=col_chart_yellow+[0.5], alpha=1.0)
cmapB2W = custom_cmap(['black', 'white'], cunder=(0.0, 0.0, 1.0, 0.5), cover=(1.0, 0.0, 0.0, 0.5), cbad=col_chart_yellow+[0.5], alpha=1.0)

# "Extended" existing colormap from matplotlib, by adding specific colors for under values, over values and bad values
# # Example for cmap 'terrain'
# cmap_terrain_ext = custom_cmap([plt.get_cmap('terrain')(x) for x in np.linspace(0, 1, 256)], ncol=256, cunder='pink', cover='orange', cbad='red')
# # or:
# cmap_terrain_ext = plt.get_cmap('terrain')
# cmap_terrain_ext.set_under('pink')
# cmap_terrain_ext.set_over('orange')
# cmap_terrain_ext.set_bad('red')
# ----------------------------------------------------------------------------
#
# # Notes:
# # =====
# # To show available colormaps in matplotlib:
# plt.colormaps()
#
# # To show some colormaps (LinearSegmentedColormap) from matplotlib:
# cm_name = [name for name in plt.cm.datad.keys()]
# cmap = plt.get_cmap('ocean')  # get the colormap named 'ocean' (cm_name[105])
#
# # To get current rcParams (matplotlib)
# import matplotlib
# matplotlib.rcParams
#
# # Useful function to convert color from/to rgb[a]/hex:
# import matplotlib
# matplotlib.mcolors.to_rgb()
# matplotlib.mcolors.to_rgba()
# matplotlib.mcolors.to_hex()
# matplotlib.mcolors.to_hex(,keep_alpha=True)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.customcolors' example:")
    import matplotlib.pyplot as plt

    # Plot a function of two variable in a given domain
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
    xx,yy = np.meshgrid(x,y)

    # Set the function values
    zz = xx**2 + yy**2 - 2
    zz[np.where(zz < -1.7)] = np.nan

    # Specify some points in the grid
    px = np.arange(xmin,xmax,.1)
    py = np.zeros(len(px))
    pz = px**2 + py**2 - 2
    pz[np.where(pz < -1.7)] = np.nan

    # Set min and max value to be displayed
    vmin, vmax = -1.0, 3.0

    # Create a custom colormap
    my_cmap = custom_cmap(('blue', 'white', 'red'), vseq=(vmin,0,vmax),
                          cunder='cyan', cover='violet', cbad='gray', alpha=.3)

    # Display
    fig, ax = plt.subplots(2,2,figsize=(16,10))
    # --- 1st plot ---
    cax = ax[0,0]
    im_plot = cax.imshow(zz, cmap=cmap1, vmin=vmin, vmax=vmax, origin='lower',
                         extent=[xmin, xmax, ymin, ymax], interpolation='none')
    cax.set_xlim(xmin, xmax)
    cax.set_ylim(ymin, ymax)
    cax.grid()
    cax.set_xlabel("x-axis")
    cax.set_xticks([-1, 1])
    cax.set_xticklabels(['x=-1', 'x=1'], fontsize=8)

    # cbarShrink = 0.9
    # plt.colorbar(extend='both', shrink=cbarShrink, aspect=20*cbarShrink)
    add_colorbar(im_plot, extend='both')

    # add points
    col = [0 for i in range(len(pz))]
    for i in range(len(pz)):
        if ~ np.isnan(pz[i]):
            col[i] = cmap1((pz[i]-vmin)/(vmax-vmin))
        else:
            col[i] = mcolors.ColorConverter().to_rgba(cbad1)

    cax.scatter(px, py, marker='o', s=50, edgecolor='black', color=col)
    cax.set_title('colormap: cmap1')

    # --- 2nd plot ---
    plt.subplot(2,2,2)
    im_plot = plt.imshow(zz,cmap=my_cmap, vmin=vmin, vmax=vmax, origin='lower',
                         extent=[xmin, xmax, ymin, ymax], interpolation='none')
    plt.grid()

    cbar = add_colorbar(im_plot, ticks=[vmin,vmax])
    # cbar.ax.set_yticklabels(cbar.get_ticks(), fontsize=16)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.ax.set_yticklabels([f'min={vmin}', 0, f'max={vmax}'], fontsize=16)

    col = [0 for i in range(len(pz))]
    for i in range(len(pz)):
        if not np.isnan(pz[i]):
            col[i] = my_cmap((pz[i]-vmin)/(vmax-vmin))
        else:
            col[i] = mcolors.ColorConverter().to_rgba('gray')

    # add points
    plt.scatter(px, py, marker='o', s=50, edgecolor='black', color=col)

    plt.title('colormap: my_cmap')

    # --- 3rd plot ---
    plt.subplot(2,2,3)
    im_plot = plt.imshow(zz, cmap=custom_cmap(col_chart_list, ncol=len(col_chart_list)),
                         vmin=vmin, vmax=vmax, origin='lower',
                         extent=[xmin,xmax,ymin,ymax], interpolation='none')
    plt.grid()
    # plt.colorbar(shrink=cbarShrink, aspect=20*cbarShrink)
    add_colorbar(im_plot)

    #col = custom_cmap(col_chart_list, ncol=len(col_chart_list))((pz-vmin)/(vmax-vmin))
    col = [0 for i in range(len(pz))]
    for i in range(len(pz)):
        if ~ np.isnan(pz[i]):
            col[i] = custom_cmap(col_chart_list, ncol=len(col_chart_list))((pz[i]-vmin)/(vmax-vmin))
        else:
            col[i] = mcolors.ColorConverter().to_rgba('white')

    plt.scatter(px, py, marker='o', s=50, edgecolor='black', color=col)

    plt.title('colormap: col_chart_list')

    # --- 4th plot ---
    plt.subplot(2,2,4)
    im_plot = plt.imshow(zz, cmap=custom_cmap(col_chart_list_s,
                                              ncol=len(col_chart_list_s)),
                         vmin=vmin, vmax=vmax, origin='lower',
                         extent=[xmin, xmax, ymin, ymax], interpolation='none')
    plt.grid()
    # plt.colorbar(shrink=cbarShrink, aspect=20*cbarShrink)
    add_colorbar(im_plot)

    #col = custom_cmap(col_chart_list_s, ncol=len(col_chart_list_s))((pz-vmin)/(vmax-vmin))
    col = [0 for i in range(len(pz))]
    for i in range(len(pz)):
        if ~ np.isnan(pz[i]):
            col[i] = custom_cmap(col_chart_list_s, ncol=len(col_chart_list_s))((pz[i]-vmin)/(vmax-vmin))
        else:
            col[i] = mcolors.ColorConverter().to_rgba('white')

    plt.scatter(px, py, marker='o', s=50, edgecolor='black', color=col)

    plt.title('colormap: col_chart_list_s')

    # main title
    plt.suptitle(r'$z=x^2 + y^2 - 2$', fontsize=24)

    # plt.tight_layout()

    # fig.show()
    plt.show()

    a = input("Press enter to continue...")
