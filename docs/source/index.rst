.. geone documentation master file, created by
   sphinx-quickstart on Mon Sep  9 13:31:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GEONE's documentation
=====================

GEONE is a Python3 package providing a set of tools for geostatistical modeling, comprising:

- a DEESSE wrapper for multiple-point statistics (MPS) simulation
- geostatistical tools based on two-point statistics
- miscellaneous algorithms based on random processes

Github
======
Github repository of GEONE : `<https://github.com/randlab/geone>`_.

.. note::

    DEESSE and some other geostatistical tools (GEOSCLASSIC) provided by GEONE are compiled in C for windows, linux and mac, and for python 3.7 to 3.12 (from python 3.8 for mac). Hence, DEESSE and GEOSCLASSIC wrappers use pre-compiled C libraries. The installation process detects the operating system and the python version, ensuring the installation of the right libraries. Note that for linux, libraries depending on the library GLIBC 2.35 or GLIBC 2.27 are provided, hence the library GLIBC of your OS has to be compatible with one of those versions to ensure proper operation of GEONE.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    install.rst
    descr.rst
    api.rst
    examples_list.rst
    examples_gallery.rst
    ref.rst
    LICENSE.rst

Authors
=======

GEONE is developed by 
`Julien Straubhaar <https://www.unine.ch/philippe.renard/home/the-team/julien-straubhaar.html>`_ 
and `Philippe Renard <https://www.unine.ch/philippe.renard/home/the-team/philippe-renard.html>`_.

License
=======

See :ref:`LICENSE <LICENSE_file>` file.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
