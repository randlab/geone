Installation
************

GEONE relies on pre-compiled C libraries (DEESSE and GEOSCLASSIC core).

.. note::
    GEONE is available on:
    
    - `PyPI <https://pypi.org/project/geone>`_ (The Python Package Index)
    - `Github repository <https://github.com/randlab/geone>`_

    GEONE relies on pre-compiled C libraries (DEESSE and GEOSCLASSIC core), available for:

    - linux (x86_64 with GLIBC 2.35 or GLIBC 2.27) and python 3.9 to 3.13
    - mac (x86_64 or arm64) and python 3.9 to 3.13
    - windows and python 3.9 to 3.13

Installation from `PyPI <https://pypi.org/project/geone>`_
-----------------------------------------------------------

In a terminal type::

    pip install geone

Or, equivalently: `python -m pip install geone`.


Installation from the `Github repository <https://github.com/randlab/geone>`_
-----------------------------------------------------------------------------

In a terminal, change directory where to download GEONE, and type::

    git clone https://github.com/randlab/geone.git
    cd geone
    pip install .

.. tip::
    Use `pip install . --verbose` or `pip install . -v` for printing (more) messages during the installation.

Alternatively:

- Instead of `git clone ...`, you can download GEONE from the `Github repository <https://github.com/randlab/geone>`_: click on the green button "code" and choose "Download ZIP". 
- Then, unzip the archive on your computer.
- Finally, in a terminal, go into the unzipped directory, and type `pip install .`

.. warning::
    If the installation has been done from github, do not launch python from the directory containing the downloaded sources and where the installation has been done (with `pip`), otherwise `import geone` will fail.

Requirements
------------

The following python packages are used by GEONE:

- `matplotlib`
- `multiprocessing` (for parallel processes)
- `numpy`
- `pandas`
- `pyvista`
- `scipy`

.. note::
    `numpy` version >= 2 is used.

.. note::
    The package `ipykernel` is required to run the notebooks.

GEONE has been tested with the following settings:

- platform_system : Linux
- python version  : sys.version_info(major=3, minor=13, micro=7, releaselevel='final', serial=0)
- glibc_str       : glibc 2.35
- glibc_version   : (2, 35)
- machine         : x86_64
- matplotlib.__version__ = 3.10.7
- numpy.__version__      = 2.3.3
- pandas.__version__     = 2.3.3
- pyvista.__version__    = 0.46.3
- scipy.__version__      = 1.16.2

Removing GEONE
--------------
In a terminal type::

    pip uninstall -y geone

.. note::
    First remove the directory 'geone.egg-info' from the current directory (if present).
