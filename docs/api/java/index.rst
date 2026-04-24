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

    import dev.vortex.api.File;
    import dev.vortex.api.Array;

    // Open a Vortex file
    File vortexFile = File.open("path/to/file.vortex");

    // Read arrays from the file
    Array array = vortexFile.readArray();

    // Work with the array data
    System.out.println("Array length: " + array.getLength());
