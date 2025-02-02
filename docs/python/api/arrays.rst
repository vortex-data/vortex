Arrays
======

The base class for all Vortex arrays is :class:`vortex.Array`.
This class holds the tree of array definitions and buffers that make up the array and can be passed into compute
functions, serialized, and otherwise manipulated as a generic array.

There are two ways of "downcasting" an array for more specific access patterns:

1. Into an encoding-specific array, like `vortex.BitPackedArray`.vortex.
2. Into a type-specific array, like `vortex.array.BoolTypeArray`.

Be careful to note that :class:`vortex.BoolArray` represents an array that stores physical data
 as a bit-buffer of booleans, vs `vortex.array.BoolTypeArray` which represents any array that has a logical
 type of boolean.

Factory Functions
-----------------

.. autofunction:: vortex.array


Base Class
----------

.. autoclass:: vortex.Array
   :members:
   :special-members: __len__


Builtin Encodings
-----------------

.. autoclass:: vortex.ChunkedArray
   :members:


.. autoclass:: vortex.ConstantArray
   :members:


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
