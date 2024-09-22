Installation
************

Installation from `PyPI <https://pypi.org/>`_ (The Python Package Index)
------------------------------------------------------------------------

In a terminal type::

    pip install geone

(or `python -m pip install geone`).

Installation from the `Github repository <https://github.com/randlab/geone>`_
-----------------------------------------------------------------------------

In a terminal, change directory where to download GEONE, and type::

    git clone https://github.com/randlab/geone.git
    cd geone
    pip install .

*Note:* use `pip install . --verbose` or `pip install . -v` for printing (more) messages during the installation.

Alternatively:

- Instead of `git clone ...`, you can download GEONE from the `Github repository <https://github.com/randlab/geone>`_: click on the green button "code" and choose "Download ZIP". 
- Then, unzip the archive on your computer
- Finally, in a terminal, go into the unzipped directory, and type `pip install .`

**Using GEONE - Important note**

If the installation has been done from github, do not launch python from the directory containing the downloaded sources and where the installation has been done (with `pip`), otherwise `import geone` will fail.

Requirements
------------
The following python packages are used by GEONE (tested on python 3.11.5):

- matplotlib (3.8.1)
- multiprocessing (for parallel processes)
- numpy (tested with version 1.26.0)
- pandas (tested with version 2.1.2)
- pyvista (tested with version 0.42.3)
- scipy (tested with version 1.11.3)

.. warning::

    numpy version **less than 2.** is required

Removing GEONE
--------------
In a terminal type::

    pip uninstall -y geone

*Note: first remove the directory 'geone.egg-info' from the current directory (if present).*
