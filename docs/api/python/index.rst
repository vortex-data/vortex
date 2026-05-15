Python API
==========

The Vortex Python API provides a Pythonic interface to the Vortex library via PyO3 bindings. It supports reading and
writing Vortex files, compressing data, and integrating with the broader Python data ecosystem including PyArrow,
Pandas, Polars, DuckDB, and Ray.

Installation
------------

.. code-block:: bash

    pip install vortex-data

Optional integrations can be installed as extras:

.. code-block:: bash

    pip install vortex-data[polars,pandas,numpy,duckdb,ray]


Compatibility
-------------

The Python bindings require Python 3.11 or newer. Pre-built wheels are available for:

* x86_64 Linux
* ARM64 Linux
* Apple Silicon macOS

They support any Linux distribution with a GLIBC version >= 2.17. This includes

* Amazon Linux 2 or newer
* Ubuntu 14.04 or newer


Usage Example
-------------

Here's a basic example of using the Vortex Python API to write and read a Vortex file:

.. code-block:: python

    import vortex

    # Write a Vortex file from a PyArrow table
    vortex.io.write_path(my_table, "data.vortex")

    # Read a Vortex file
    dataset = vortex.dataset("data.vortex")
    table = dataset.to_arrow()


API Reference
-------------

.. toctree::
   :maxdepth: 5

   dtypes
   scalars
   arrays
   expr
   compress
   registry
   io
   store
   dataset
   runtime
   type_aliases
