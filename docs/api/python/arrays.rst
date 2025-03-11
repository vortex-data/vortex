Arrays
======

The base class for all Vortex arrays is :class:`vortex.Array`.
This class holds the tree of array definitions and buffers that make up the array and can be passed into compute
functions, serialized, and otherwise manipulated as a generic array.

Factory Functions
-----------------

.. autofunction:: vortex.array


Base Class
----------

.. autoclass:: vortex.Array
    :members:
    :special-members: __len__


Canonical Encodings
-------------------

Each :class:`~vortex.DType` has a corresponding canonical encoding. These encodings represent the uncompressed version
of the array, and are also zero-copy to Apache Arrow.

.. autoclass:: vortex.NullArray
     :members:


.. autoclass:: vortex.BoolArray
     :members:


.. autoclass:: vortex.PrimitiveArray
     :members:


.. autoclass:: vortex.VarBinArray
    :members:


.. autoclass:: vortex.VarBinViewArray
    :members:


.. autoclass:: vortex.StructArray
    :members:


.. autoclass:: vortex.ListArray
    :members:


.. autoclass:: vortex.ExtensionArray
    :members:


Utility Encodings
-----------------

.. autoclass:: vortex.ChunkedArray
    :members:


.. autoclass:: vortex.ConstantArray
    :members:


.. autoclass:: vortex.ByteBoolArray
     :members:


.. autoclass:: vortex.SparseArray
    :members:


Compressed Encodings
--------------------

.. autoclass:: vortex.AlpArray
    :members:


.. autoclass:: vortex.AlpRdArray
    :members:


.. autoclass:: vortex.DateTimePartsArray
    :members:


.. autoclass:: vortex.DictArray
    :members:


.. autoclass:: vortex.FsstArray
    :members:


.. autoclass:: vortex.RunEndArray
    :members:


.. autoclass:: vortex.ZigZagArray
    :members:


.. autoclass:: vortex.FastLanesBitPackedArray
    :members:


.. autoclass:: vortex.FastLanesDeltaArray
    :members:


.. autoclass:: vortex.FastLanesFoRArray
    :members:


Pluggable Encodings
-------------------

Subclasses of :class:`~vortex.PyArray` can be used to implement custom Vortex encodings in Python. These encodings
can be registered with the :attr:`~vortex.registry` so they are available to use when reading Vortex files.

.. autoclass:: vortex.PyArray
    :members:


Registry and Serde
------------------

.. autodata:: vortex.registry

.. autoclass:: vortex.Registry
    :members:

.. autoclass:: vortex.ArrayContext
    :members:

.. autoclass:: vortex.ArrayParts
    :members:


Streams and Iterators
---------------------

.. autoclass:: vortex.ArrayIterator
    :members: