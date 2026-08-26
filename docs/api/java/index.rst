Java API
========

The Vortex Java API provides bindings for the Vortex library, enabling Java applications to work with Vortex arrays and files.

The API is split into two main components:

* **Vortex JNI**: Core JNI bindings for Vortex functionality
* **Vortex Spark**: Apache Spark integration for reading Vortex files

.. raw:: html

   <div class="api-links">
   <h3>API Documentation</h3>
   <ul>
     <li><a href="../../_static/vortex-jni/index.html">Vortex JNI API</a> - Core Java bindings for Vortex</li>
     <li><a href="../../_static/vortex-spark/index.html">Vortex Spark API</a> - Apache Spark integration</li>
   </ul>
   </div>

Installation
------------

The Java API can be included in your project using Gradle or Maven. Please refer to the main documentation for detailed installation instructions.


Compatibility
-------------

The Java bindings are supported on the following architectures:

* x86_64 Linux
* ARM64 Linux
* Apple Silicon macOS

They support any Linux distribution with a GLIBC version >= 2.31. This includes

* Amazon Linux 2022 or newer
* Ubuntu 20.04 or newer


Usage Example
-------------

Here's a basic example of using the Vortex Java API to read a Vortex file:

.. code-block:: java

    import dev.vortex.api.DataSource;
    import dev.vortex.api.Partition;
    import dev.vortex.api.Scan;
    import dev.vortex.api.ScanOptions;
    import dev.vortex.api.Session;
    import dev.vortex.arrow.ArrowAllocation;
    import org.apache.arrow.memory.BufferAllocator;
    import org.apache.arrow.vector.VectorSchemaRoot;
    import org.apache.arrow.vector.ipc.ArrowReader;

    BufferAllocator allocator = ArrowAllocation.rootAllocator();
    Session session = Session.create();
    DataSource source = DataSource.open(session, "path/to/file.vortex");

    // A scan yields one partition per chunk of the file.
    Scan scan = source.scan(ScanOptions.of());
    while (scan.hasNext()) {
        Partition partition = scan.next();
        try (ArrowReader reader = partition.scanArrow(allocator)) {
            while (reader.loadNextBatch()) {
                VectorSchemaRoot batch = reader.getVectorSchemaRoot();
                System.out.println("read " + batch.getRowCount() + " rows");
            }
        }
    }

Data crosses the JNI boundary as Arrow record batches, so the buffers stay in native memory and
are read from Java through the Arrow C Data Interface.
