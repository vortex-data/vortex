C++ API
=======

The Vortex C++ API allows you to read and write ``.vortex`` files directly or
via an Arrow compatibility layer like `nanoarrow
<https://arrow.apache.org/nanoarrow/>`_. The only dependency apart from Vortex
is ``nanoarrow``.

.. note::
   C++ API is a work in progress. Please reach out to us if you are interested
   in using Vortex from C++ or you want a feature not covered yet e.g.
   extension support.

Installation
------------

We don't provide prebuilt library files (yet) so you will need to build Vortex
from source, and for that you will need:

- C++20,
- Rust toolchain,
- and CMake 3.10.

.. code-block:: bash

    git clone --depth 1 https://github.com/vortex-data/vortex
    cd vortex
    cargo build --release -p vortex-ffi

    cmake -S lang/cpp -Bbuild -DCMAKE_BUILD_TYPE=Release
    # To build the examples, pass -DBUILD_EXAMPLES=1
    # cmake -S lang/cpp -Bbuild -DBUILD_EXAMPLES=1

    cmake --build build -j

This produces a shared and a static library which you can use directly or via

.. code-block:: cmake

    # static library
    target_link_libraries(target PRIVATE vortex_cxx)
    # shared library
    target_link_libraries(target PRIVATE vortex_cxx_shared)

Have a look at the `examples
<https://github.com/vortex-data/vortex/tree/develop/lang/cpp/examples>`_
directory as well.

Reading files
-------------

Full source code for this example is `reader.cpp
<https://github.com/vortex-data/vortex/tree/develop/lang/cpp/examples/reader.cpp>`_
. For brevity we omit ``main()``, system includes and ``using`` directives.

Assuming you have Vortex files ``people0``, ``people1``, and ``me`` in a local folder,
each containing U8 column "age" and U16 column "height", this is how you
print all ages for specific heights:

.. literalinclude:: ../../../lang/cpp/examples/reader.cpp
   :language: cpp
   :start-after: docs:begin:example
   :end-before: docs:end:example
   :dedent: 4

DataSource and Scan
^^^^^^^^^^^^^^^^^^^

First, you need to create a Vortex session, which does extension bookkeeping and
may also hold metadata like object store credentials (we'll get back to it
later). Further, a DataSource provides a view over multiple files which may also
be remote. You can specify globs for every item as it's shown.

Once you have a DataSource, you can create Scans out of it. A Scan is a single
traversal of a DataSource which projects columns and filters on them. Our Scan
is consumed by following calls so it needs to be mutable. ScanOptions, which is
passed to Scan, is a simple C++ aggregate so you can initialize any fields you
want or avoid them altogether (``auto scan = ds.scan()``).

Expressions
^^^^^^^^^^^

If you omit ``.projection`` all columns are read as part of a scan. If we wanted
to return only the "age" column we'd write ``.projection = col("age")``. We pass
an Expression to a filter which returns false for some "height" values, and the
scan filters them out.

Two additional things to look for in ``.filter``: first, we allow overloading
Expression operators which produce Expressions themselves à la Eigen. This is
opt-in via ``using namespace vortex::expr::ops``. If you prefer, you can use
``eq(col("height"), lit<uint16_t>(180))`` instead. The second thing is that we
explicitly pass the type to the ``lit`` expression, which creates a literal
constant. We don't do any type coercion in Vortex so if you were to write
``lit(180)``, C++ would likely deduce the type to ``int`` and fail at runtime.

Once we're done with the scan, we need to consume the data it provides. Vortex
has Arrow interoperability, but now let's focus on Partitions and Arrays.

Partitions and Arrays
^^^^^^^^^^^^^^^^^^^^^

A Partition is an independent unit of work. Assuming your Vortex file
processing will likely be multithreaded, each thread can pull Partitions from
a Scan and handle them in parallel. For simplicity we don't cover it here, but
look for the next sections! Each Partition produces Arrays which are batches of
rows and columns.

.. note::
   Arrays, DataTypes, and Expressions are reference-counted so copying them is
   cheap.

Once we have an Array, we can get access to its values. First, we need to
extract the "age" field.

.. note::
   Default projection returns the root Array which is a Struct so to get
   a field from it we need to use ``field()``. If we were to use ``col()`` in
   projection, the array we got would be "age" column already, so we wouldn't
   need to use ``field()``.


Then we want to get access to the raw bytes. However,
Array likely references compressed data, so now we need to learn about
canonicalization.

Canonicalization
^^^^^^^^^^^^^^^^

Vortex files hold trees of compressed data where each node is a specific
encoding (zstd, FSST, delta) over another node. This is good for performance
because we defer decompression and we can also pass compressed data directly to
other systems. Say, if a reader knows how to deal with bitpacked integers but
not RLE and we have ``RLE(Bitpacked(U64))``, we can decompress just RLE and
pass bitpacked array directly to the reader.

In this example we want to remove all encodings and uncompress all data fully.
This form is called a canonical Array, and the process is canonicalization.

When we request ``.values(session)``, we canonicalize the Array and get a
PrimitiveView because we already know the type of the column. As PrimitiveView
holds uncompressed data, we can get raw ``uint64_t`` numbers by calling
``.values()`` on the view.

Writing files
-------------

Now let's write the first files to be read by our previous example.
Source code for this example is `writer.cpp
<https://github.com/vortex-data/vortex/tree/develop/lang/cpp/examples/writer.cpp>`_.

.. literalinclude:: ../../../lang/cpp/examples/writer.cpp
   :language: cpp
   :start-after: docs:begin:example
   :end-before: docs:end:example
   :dedent: 4

DataType
^^^^^^^^

First, you want to create a schema. DataType is a logical representation of
Vortex's type system. "Logical" as opposed to "physical" means DataType doesn't
know what encodings the data use. As another example, Arrow and Parquet's types
are physical. There's a convenience function ``struct_`` which creates a
``DataTypeVariant::Struct`` with fields passed as a ``initializer_list``.

Each DataType has a notion of nullability: whether some items in the row can be
invalid (Vortex uses the terms "null" and "invalid" interchangeably). DataTypes
are non-nullable by default, so "age" column is not nullable, but "height" is.

Once we create Arrays for columns we can merge them into a Struct Array with
``make_struct``, and we need to say a word about Validity.

Nullability and Validity
^^^^^^^^^^^^^^^^^^^^^^^^

Nullability (and ``Nullable`` flag) is a type-system note that we `may` have
invalid elements in the column. A non-``Nullable`` column can't have nulls in
it, but a ``Nullable`` column's items may be all valid as well. Validity, on the
other hand, tells us whether we `do` have nulls in a particular Array.

"height" in the first Array is AllValid which means in this Array we don't have
invalid items. For the second Array we reuse age by copying it (which is a
reference clone so it's cheap). But for "height" we say Validity is determined
by another Array. This array consists of Bools and we obtained it by applying a
comparison Expression to the "age" column of the first array. So, "height" field
in ``array2`` will have null/invalid values if "age"'s item at the same index
was less than 11.

Applying Expressions
^^^^^^^^^^^^^^^^^^^^

One important thing to know is that applying an Expression is a constant-time
operation, and the returned array contains the original values along with some
meta-information about the expression. Real application happens when you execute
the array. One example of executing the array is canonicalization, which in this
example happens later when we write both arrays to the file.

Writing to a file
^^^^^^^^^^^^^^^^^

Once we're done with the arrays, we need to initialize the Writer and push
arrays into it. ``finish()`` call writes the file footer. If you don't call it and
let Writer go out of scope, the file will be visibly corrupted.

Now you can build the example and read back the generated files:

.. code-block::

    ./build/examples/writer people0.vortex
    ./build/examples/writer people1.vortex
    ./build/examples/writer me.vortex
    ./build/examples/reader
