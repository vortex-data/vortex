Data Types
==========

The logical types of the elements of an Array. Each logical type is implemented by a variety of
Array encodings which describe both a representation-as-bytes as well as how to apply operations on
that representation.

Factory Functions
-----------------

.. autofunction:: vortex.null
.. autofunction:: vortex.bool_
.. autofunction:: vortex.float_
.. autofunction:: vortex.int_
.. autofunction:: vortex.uint
.. autofunction:: vortex.utf8
.. autofunction:: vortex.binary
.. autofunction:: vortex.struct
.. autofunction:: vortex.list_

Base Class
----------

Do not instantiate these classes directly. Instead, call one of the factory functions above.

.. autoclass:: vortex.DType
    :members:


DType Classes
-------------

.. autoclass:: vortex.NullDType
    :members:


.. autoclass:: vortex.BoolDType
     :members:


.. autoclass:: vortex.PrimitiveDType
     :members:


.. autoclass:: vortex.Utf8DType
     :members:


.. autoclass:: vortex.BinaryDType
     :members:


.. autoclass:: vortex.StructDType
     :members:


.. autoclass:: vortex.ListDType
     :members:


.. autoclass:: vortex.ExtensionDType
     :members:


Primitive Types
---------------

.. autoclass:: vortex.PType
     :members:
