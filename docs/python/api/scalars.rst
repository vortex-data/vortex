Scalars
=======

A scalar is a single atomic value like the integer ``1``, the string ``"hello"``, or the structure
``{"age": 55, "name": "Angela"}``. The :meth:`.Array.scalar_at` method
returns a native Python value when the cost of doing so is small. However, for larger values like
binary data, UTF-8 strings, variable-length lists, and structures, Vortex returns a zero-copy *view*
of the Array data. The ``into_python`` method of each view will copy the scalar into a native Python
value.

Factory Function
----------------

.. autofunction:: vortex.scalar


Base Class
----------

.. autoclass:: vortex.Scalar
   :members:


Scalar Classes
--------------

.. autoclass:: vortex.NullScalar
   :members:

.. autoclass:: vortex.BoolScalar
   :members:

.. autoclass:: vortex.PrimitiveScalar
   :members:

.. autoclass:: vortex.Utf8Scalar
   :members:

.. autoclass:: vortex.BinaryScalar
   :members:

.. autoclass:: vortex.StructScalar
   :members:

.. autoclass:: vortex.ListScalar
   :members:

.. autoclass:: vortex.ExtensionScalar
   :members:
