DTypes
======

.. c:autotype:: vx_dtype
   :file: vortex.h

.. c:autofunction:: vx_dtype_clone
    :file: vortex.h

.. c:autofunction:: vx_dtype_free
    :file: vortex.h

Factories
---------

.. c:autofunction:: vx_dtype_new_null
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_bool
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_primitive
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_decimal
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_utf8
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_binary
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_struct
   :file: vortex.h

.. c:autofunction:: vx_dtype_new_list
   :file: vortex.h

Properties
----------

.. c:autoenum:: vx_dtype_variant
    :file: vortex.h
    :members:

.. c:autofunction:: vx_dtype_get_variant
    :file: vortex.h

.. c:autofunction:: vx_dtype_is_nullable
    :file: vortex.h

.. c:autofunction:: vx_dtype_primitive_ptype
    :file: vortex.h

.. c:autofunction:: vx_dtype_decimal_precision
    :file: vortex.h

.. c:autofunction:: vx_dtype_decimal_scale
    :file: vortex.h

.. c:autofunction:: vx_dtype_struct_dtype
    :file: vortex.h

.. c:autofunction:: vx_dtype_list_element
    :file: vortex.h

PType
-----

.. c:autoenum:: vx_ptype
    :file: vortex.h
    :members:

Struct Fields
-------------

.. c:autotype:: vx_struct_fields
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_clone
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_free
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_nfields
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_field_dtype
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_field_name
    :file: vortex.h

Struct Fields Builder
^^^^^^^^^^^^^^^^^^^^^

.. c:autotype:: vx_struct_fields_builder
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_builder_free
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_builder_new
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_builder_add_field
    :file: vortex.h

.. c:autofunction:: vx_struct_fields_builder_finalize
    :file: vortex.h
