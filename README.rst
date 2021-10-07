=======
cubiints
=======
|Doc Status|
|Pypi Version|
|Build Version|
|Python Versions|
|Project License|

A tool for searching optimal solution intervals for the radio intereferometric calibration.

.. Main website: https://aimfast.readthedocs.io

==============
Introduction
==============

.. Image fidelity is a measure of the accuracy of the reconstructed sky brightness distribution. A related metric, dynamic range, is a measure of the degree to which imaging artifacts around strong sources are suppressed, which in turn implies a higher fidelity of the on-source reconstruction. Moreover, the choice of image reconstruction algorithm also affects the correctness of the on-source brightness distribution. For high dynamic ranges with wide bandwidths, algorithms that model the sky spectrum as well as the average intensity can yield more accurate reconstructions.

==============
Installation
==============
Installation from source_,
working directory where source is checked out

.. code-block:: bash
  
    $ pip install .

Pre release from git 

.. code-block:: bash
  
    $ pip install -e https://github.com/ulricharmel/cubiInts@v0.2-alpha

To run and see options

.. code-block:: bash

    $ cubiints --help 

=======
License
=======

This project is licensed under the GNU General Public License v3.0 - see license_ for details.

