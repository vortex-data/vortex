Input and Output
================

Vortex arrays support reading and writing to local and remote file systems, including plain-old
HTTP, S3, Google Cloud Storage, and Azure Blob Storage.

.. autosummary::
   :nosignatures:

   ~vortex.open
   ~vortex.open_readable
   ~vortex.io.ReadAt
   ~vortex.io.ReadBytesAt
   ~vortex.SegmentCache
   ~vortex.VortexFile
   ~vortex.file.Footer
   ~vortex.RepeatedScan
   ~vortex.io.read_url
   ~vortex.io.write

.. raw:: html

   <hr>

.. autofunction:: vortex.open

.. autofunction:: vortex.open_readable

.. autoclass:: vortex.io.ReadAt
   :members:

.. autoclass:: vortex.io.ReadBytesAt
   :members:

.. autoclass:: vortex.SegmentCache
   :members:

.. autoclass:: vortex.VortexFile
   :members:

.. autoclass:: vortex.file.Footer
   :members:

.. autoclass:: vortex.RepeatedScan
   :members:

.. automodule:: vortex.io
    :members:
    :imported-members:
    :exclude-members: ReadAt, ReadBytesAt

