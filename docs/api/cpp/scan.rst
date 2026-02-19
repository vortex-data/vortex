Scanning
========

The scan API provides a builder pattern for reading data from a Vortex file with optional
filter, projection, row range, and limit pushdowns. The resulting stream exposes the
Arrow C Data Interface (``ArrowArrayStream``).

ScanBuilder
-----------

.. doxygenclass:: vortex::ScanBuilder
   :members:

StreamDriver
------------

.. doxygenclass:: vortex::StreamDriver
   :members:
