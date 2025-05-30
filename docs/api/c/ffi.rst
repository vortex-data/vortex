Vortex FFI API
==============

.. warning::
    This API should be considered entirely unstable. It _will_ change. Please reach out if a stable
    FFI API is important for your use-case and we can accelerate the process of stabilizing it.

Sessions
--------

While not all parts of Vortex require a session, many do. A Vortex session object holds registries of extensible
types, such as array encodings, layout encodings, extension dtypes, compute functions, and more.

.. c:autotype:: vx_session
   :file: vortex.h

.. c:autofunction:: vx_session_new
   :file: vortex.h

.. c:autofunction:: vx_session_free
   :file: vortex.h

DTypes
------

.. c:autotype:: vx_dtype
   :file: vortex.h

.. c:autofunction:: vx_dtype_new
   :file: vortex.h

.. c:autofunction:: vx_dtype_free
    :file: vortex.h

Arrays
------

.. c:autotype:: vx_array
   :file: vortex.h

.. c:autofunction:: vx_array_dtype
   :file: vortex.h

Errors
------

.. c:autotype:: vx_error
   :file: vortex.h

.. c:autofunction:: vx_error_get_message
   :file: vortex.h

.. c:autofunction:: vx_error_get_code
   :file: vortex.h

Logging
-------

.. c:autofunction:: vx_set_log_level
   :file: vortex.h

.. c:autoenum:: vx_log_level
    :file: vortex.h
    :members:
