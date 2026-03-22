C API
=====

The Vortex C API provides a low-level FFI interface to the Vortex library. It is the foundation for
other language bindings (including C++) and is suitable for embedding Vortex into C applications or
building higher-level wrappers.

.. warning::
    This API should be considered entirely unstable. It *will* change. Please reach out if a stable
    FFI API is important for your use-case and we can accelerate the process of stabilizing it.


Installation
------------

The C API is provided as a static or shared library built from the ``vortex-ffi`` crate. To build it:

.. code-block:: bash

    cargo build -p vortex-ffi

The generated header file ``vortex.h`` and compiled library can then be linked into your C project.


Compatibility
-------------

The C bindings are supported on the following architectures:

* x86_64 Linux
* ARM64 Linux
* Apple Silicon macOS

They support any Linux distribution with a GLIBC version >= 2.31. This includes

* Amazon Linux 2022 or newer
* Ubuntu 20.04 or newer


API Reference
-------------

.. toctree::
   :maxdepth: 1

   dtypes
   arrays

Session
-------

While not all parts of Vortex require a session, many do. A Vortex session object holds registries of extensible
types, such as array encodings, layout encodings, extension dtypes, compute functions, and more.

.. c:autotype:: vx_session
   :file: vortex.h

.. c:autofunction:: vx_session_free
   :file: vortex.h

.. c:autofunction:: vx_session_new
   :file: vortex.h

Logging
-------

.. c:autofunction:: vx_set_log_level
   :file: vortex.h

.. c:autoenum:: vx_log_level
    :file: vortex.h
    :members:

Errors
------

Errors are passed out of many function in the Vortex C API. Each time they will be heap-allocated and the caller is
responsible for freeing them.

.. c:autotype:: vx_error
   :file: vortex.h

.. c:autofunction:: vx_error_free
   :file: vortex.h

.. c:autofunction:: vx_error_get_message
   :file: vortex.h

Strings
-------

Vortex strings wrap a Rust `Arc<str>`, and therefore are reference-counted, UTF-8 encoded, and not null-terminated.

.. c:autotype:: vx_string
   :file: vortex.h

.. c:autofunction:: vx_string_clone
   :file: vortex.h

.. c:autofunction:: vx_string_free
   :file: vortex.h

.. c:autofunction:: vx_string_new
   :file: vortex.h

.. c:autofunction:: vx_string_new_from_cstr
   :file: vortex.h

.. c:autofunction:: vx_string_len
    :file: vortex.h

.. c:autofunction:: vx_string_ptr
    :file: vortex.h
