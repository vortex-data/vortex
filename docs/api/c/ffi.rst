Vortex FFI API
==============

.. warning::
    This API should be considered entirely unstable. It _will_ change. Please reach out if a stable
    FFI API is important for your use-case and we can accelerate the process of stabilizing it.

Session
-------

While not all parts of Vortex require a session, many do. A Vortex session object holds registries of extensible
types, such as array encodings, layout encodings, extension dtypes, compute functions, and more.

.. c:autotype:: vx_session
   :file: vortex.h

.. c:autofunction:: vx_session_new
   :file: vortex.h

.. c:autofunction:: vx_session_free
   :file: vortex.h

DType
-----

.. c:autotype:: vx_dtype
   :file: vortex.h

.. c:autofunction:: vx_dtype_clone
    :file: vortex.h

.. c:autofunction:: vx_dtype_free
    :file: vortex.h

Factories
^^^^^^^^^

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
^^^^^^^^^^

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
^^^^^

.. c:autoenum:: vx_ptype
    :file: vortex.h
    :members:

Struct DType
^^^^^^^^^^^^

.. c:autotype:: vx_struct_dtype
    :file: vortex.h

.. c:autofunction:: vx_struct_dtype_clone
    :file: vortex.h

.. c:autofunction:: vx_struct_dtype_free
    :file: vortex.h

Struct DType Builder
^^^^^^^^^^^^^^^^^^^^

.. c:autotype:: vx_struct_dtype_builder
    :file: vortex.h

.. c:autofunction:: vx_struct_dtype_builder_new
    :file: vortex.h

.. c:autofunction:: vx_struct_dtype_builder_add_field
    :file: vortex.h

.. c:autofunction:: vx_struct_dtype_builder_finalize
    :file: vortex.h

Array
-----

.. c:autotype:: vx_array
   :file: vortex.h

.. c:autofunction:: vx_array_dtype
   :file: vortex.h

Error
-----

.. c:autotype:: vx_error
   :file: vortex.h

.. c:autofunction:: vx_error_get_message
   :file: vortex.h

.. c:autofunction:: vx_error_get_code
   :file: vortex.h

String
------

.. c:autotype:: vx_string
   :file: vortex.h

.. c:autofunction:: vx_string_new
   :file: vortex.h

.. c:autofunction:: vx_string_new_from_cstr
   :file: vortex.h

.. c:autofunction:: vx_string_len
    :file: vortex.h

.. c:autofunction:: vx_string_ptr
    :file: vortex.h

.. c:autofunction:: vx_string_free
   :file: vortex.h

Logging
-------

.. c:autofunction:: vx_set_log_level
   :file: vortex.h

.. c:autoenum:: vx_log_level
    :file: vortex.h
    :members:
