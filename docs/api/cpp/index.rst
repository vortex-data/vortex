C++ API
=======

The Vortex C++ API provides an idiomatic C++ wrapper around the Vortex C FFI, built using
`cxx <https://cxx.rs/>`_. It currently supports reading and writing Vortex files and integrates
with the Arrow C Data Interface via `nanoarrow <https://arrow.apache.org/nanoarrow/>`_.

In the future we will expand the C++ API to cover Vortex's plugin and extension points. Please
reach out if you are interested in extending Vortex from C++ so we can prioritize these features.

.. note::
    Both the C++ API and this documentation are a work in progress. The API surface may change
    significantly. Please reach out if you are interested in using Vortex from C++ so we can
    prioritize stabilization.


Installation
------------

The C++ bindings are built using CMake. Requirements:

* CMake 3.22 or higher
* C++20 compatible compiler
* Rust toolchain (for building the underlying Rust library)

.. code-block:: bash

    cd vortex-cxx
    mkdir build && cd build
    cmake ..
    make -j$(nproc)


Compatibility
-------------

The C++ bindings are supported on the following architectures:

* x86_64 Linux
* ARM64 Linux
* Apple Silicon macOS

They support any Linux distribution with a GLIBC version >= 2.31. This includes

* Amazon Linux 2022 or newer
* Ubuntu 20.04 or newer


Usage Example
-------------

Here's a basic example of reading a Vortex file into an Arrow array stream:

.. code-block:: cpp

    #include "vortex/file.hpp"
    #include "vortex/scan.hpp"

    // Open a Vortex file and scan with a row limit
    auto stream = vortex::VortexFile::Open("data.vortex")
        .CreateScanBuilder()
        .WithLimit(1000)
        .IntoStream();

    // Consume the Arrow C Data stream
    ArrowArray array;
    while (stream.get_next(&stream, &array) == 0 && array.release != nullptr) {
        // Process each batch...
    }


API Reference
-------------

.. toctree::
   :maxdepth: 2

   dtypes
   scalars
   expr
   file
   scan
