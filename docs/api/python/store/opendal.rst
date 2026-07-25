=============
OpenDAL (COS)
=============

Vortex can read from and write to Tencent Cloud COS through
`OpenDAL <https://opendal.apache.org/>`_, which provides native service support.

This store is available only when Vortex is built with the ``opendal`` feature
(e.g. ``maturin develop --features opendal`` or ``cargo build -p vortex-jni --features opendal``).

.. autoclass:: vortex.store.CosStore
   :members:

Reading from COS
================

Pass a ``cos://`` URL directly. Credentials and the endpoint are picked up from the environment
variables OpenDAL's COS builder reads (``TENCENTCLOUD_SECRET_ID``, ``TENCENTCLOUD_SECRET_KEY`` and
``COS_ENDPOINT``):

.. code-block:: python

   import vortex as vx

   a = vx.io.read_url("cos://my-bucket/path/to/dataset.vortex")

Or configure explicitly with :class:`~vortex.store.CosStore` and pass it to
:func:`vortex.io.read_url` via ``store=``:

.. code-block:: python

   from vortex.io import read_url
   from vortex.store import CosStore

   store = CosStore(
       bucket="my-bucket",
       endpoint="https://cos.ap-guangzhou.myqcloud.com",
       secret_id="AKID...",
       secret_key="...",
   )

   # When `store=` is supplied, the path is a key within the store, so the scheme and
   # bucket are not part of the path passed to read_url.
   a = read_url("path/to/dataset.vortex", store=store)

Passing a store object directly
===============================

``CosStore`` is a concrete store object. You can build one once and pass it
directly to :func:`vortex.io.read_url` / :func:`vortex.io.write` via the ``store=`` argument,
exactly like the built-in S3/Azure/GCS stores:

.. code-block:: python

   from vortex.io import read_url
   from vortex.store import CosStore

   store = CosStore(
       bucket="my-bucket",
       endpoint="https://cos.ap-guangzhou.myqcloud.com",
       secret_id="AKID...",
       secret_key="...",
   )

   # When `store=` is supplied, the path is resolved as a key within the store, so the scheme
   # and bucket are not part of the path passed to read_url.
   a = read_url("path/to/dataset.vortex", store=store)
