Encodings
=========

Vortex arrays have both a logical data type and a physical encoding. Arrays in PyVortex are downcast to their
specific physical encoding where such a Python class exists, otherwise a base :class:`~vortex.Array` is used.

Each encoding-specific class may have additional methods and properties that are specific to that encoding.
To be concise, we do not show the base class methods in this encoding-specific class documentation.

.. autofunction:: vortex.compress

.. autoclass:: vortex.encoding.BoolArray
    :members:
    :show-inheritance: