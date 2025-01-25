Arrays
======

A Vortex array is a possibly compressed ordered set of homogeneously typed values. Each array has a
logical type and a physical encoding. The logical type describes the set of operations applicable to
the values of this array. The physical encoding describes how this array is realized in memory, on
disk, and over the wire and how to apply operations to that realization.

.. autosummary::
   :nosignatures:

   ~vortex.encoding.array
   ~vortex.encoding.compress
   ~vortex.encoding.Array

.. raw:: html

   <hr>

.. autofunction:: vortex.encoding.array

.. autofunction:: vortex.encoding.compress

.. autoclass:: vortex.encoding.Array
   :members:
   :special-members: __len__
