Encoded Arrays
==============

The base class for all Vortex arrays is :class:`vortex.Array`.
By default, Vortex arrays are subclassed to their type-specific array, like :class:`vortex.BoolTypeArray`.

An alternative way of accessing arrays is via their encoding-specific subclass. These classes can be constructed
by passing the array into the constructor, and then specific functionality can be accessed.

Base Class
----------

Encoded arrays are implemented as subclasses of :class:`~vortex.Array`.

.. autoclass:: vortex.Array
    :members:
    :no-index:

Canonical Encodings
-------------------

Each :class:`~vortex.DType` has a corresponding canonical encoding. These encodings represent the uncompressed version
of the array, and are also zero-copy to Apache Arrow.

.. autoclass:: vortex.NullEncoding
     :members:


.. autoclass:: vortex.BoolEncoding
     :members:


.. autoclass:: vortex.PrimitiveEncoding
     :members:


.. autoclass:: vortex.VarBinEncoding
    :members:


.. autoclass:: vortex.VarBinViewEncoding
    :members:


.. autoclass:: vortex.StructEncoding
    :members:


.. autoclass:: vortex.ListEncoding
    :members:


.. autoclass:: vortex.ExtensionEncoding
    :members:


Utility Encodings
-----------------

.. autoclass:: vortex.ChunkedEncoding
    :members:


.. autoclass:: vortex.ConstantEncoding
    :members:


.. autoclass:: vortex.SparseEncoding
    :members:


Compressed Encodings
--------------------

.. autoclass:: vortex.AlpEncoding
    :members:


.. autoclass:: vortex.AlpRdEncoding
    :members:


.. autoclass:: vortex.DateTimePartsEncoding
    :members:


.. autoclass:: vortex.DictEncoding
    :members:


.. autoclass:: vortex.FsstEncoding
    :members:


.. autoclass:: vortex.RunEndEncoding
    :members:


.. autoclass:: vortex.ZigZagEncoding
    :members:


.. autoclass:: vortex.FastLanesBitPackedEncoding
    :members:


.. autoclass:: vortex.FastLanesDeltaEncoding
    :members:


.. autoclass:: vortex.FastLanesForEncoding
    :members:
