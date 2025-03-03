Arrays
======

The base class for all Vortex arrays is :class:`vortex.Array`.
This class holds the tree of array definitions and buffers that make up the array and can be passed into compute
functions, serialized, and otherwise manipulated as a generic array.

There are two ways of "downcasting" an array for more specific access patterns:

1. Into an encoding-specific array, like :class:`vortex.FastLanesBitPackedArray`.vortex.
2. Into a type-specific array, like :class:`vortex.BoolTypeArray`.

Be careful to note that :class:`vortex.BoolArray` represents an array that stores physical data
 as a bit-buffer of booleans, vs :class:`vortex.BoolTypeArray` which represents any array that has a logical
 type of boolean.

Factory Functions
-----------------

.. autofunction:: vortex.array


Base Class
----------

.. autoclass:: vortex.Array
    :members:
    :special-members: __len__


Typed Arrays
------------

By default, the array subclass returned from PyVortex will be specific to the :class:`~vortex.DType` of the array.
These subclasses expose type-specific functionality that is more useful for the average use-case than encoding-specific
functionality.

.. autoclass:: vortex.NullTypeArray
     :members:

.. autoclass:: vortex.BoolTypeArray
     :members:

.. autoclass:: vortex.PrimitiveTypeArray
     :members:

.. autoclass:: vortex.UIntTypeArray
     :members:

.. autoclass:: vortex.UInt8TypeArray
     :members:

.. autoclass:: vortex.UInt16TypeArray
     :members:

.. autoclass:: vortex.UInt32TypeArray
     :members:

.. autoclass:: vortex.UInt64TypeArray
     :members:

.. autoclass:: vortex.IntTypeArray
     :members:

.. autoclass:: vortex.Int8TypeArray
     :members:

.. autoclass:: vortex.Int16TypeArray
     :members:

.. autoclass:: vortex.Int32TypeArray
     :members:

.. autoclass:: vortex.Int64TypeArray
     :members:

.. autoclass:: vortex.FloatTypeArray
     :members:

.. autoclass:: vortex.Float16TypeArray
     :members:

.. autoclass:: vortex.Float32TypeArray
     :members:

.. autoclass:: vortex.Float64TypeArray
     :members:

.. autoclass:: vortex.Utf8TypeArray
     :members:

.. autoclass:: vortex.BinaryTypeArray
     :members:

.. autoclass:: vortex.StructTypeArray
     :members:

.. autoclass:: vortex.ListTypeArray
     :members:

.. autoclass:: vortex.ExtensionTypeArray
     :members:


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