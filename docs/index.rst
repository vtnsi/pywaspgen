.. pywaspgen documentation master file, created by
   sphinx-quickstart on Mon Oct  6 12:30:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyWASPGEN documentation
=======================

PyWASPGEN (Python Wideband Aggregate SPectrum GENerator) is intended as a native python dataset generation tool for creating synthetic aggregate radio frequency captures for initial testing and evaluation of spectrum sensing algorithms. The data produced by this tool is particularly useful for testing signal detection algorithms (i.e. where in time and frequency signals exist in the capture) as well as signal classification algorithms (i.e. what is the signaling format of the detected signal).

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Using Pywaspgen:
   :hidden:

   building_configuration_files
   utilizing_pywaspgen

.. toctree::
   :maxdepth: 5
   :caption: Pywaspgen API:
   :hidden:

   modules/modules