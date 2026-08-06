===========================
OpenDAL (COS, OSS, GooseFS)
===========================

Vortex can read from and write to Tencent Cloud COS, Alibaba Cloud OSS, and Tencent Cloud
GooseFS through `OpenDAL <https://opendal.apache.org/>`_, which provides native service
support.

These stores are available only when Vortex is built with the ``opendal`` feature
(e.g. ``maturin develop --features opendal`` or ``cargo build -p vortex-jni --features opendal``).

.. list-table::
   :header-rows: 1

   * - Scheme
     - Service
     - Endpoint / master variable
     - Credential variables
   * - ``cos://``
     - Tencent Cloud COS
     - ``COS_ENDPOINT``
     - ``TENCENTCLOUD_SECRET_ID``, ``TENCENTCLOUD_SECRET_KEY``
   * - ``oss://``
     - Alibaba Cloud OSS
     - ``OSS_ENDPOINT``
     - ``ALIBABA_CLOUD_ACCESS_KEY_ID``, ``ALIBABA_CLOUD_ACCESS_KEY_SECRET``
   * - ``goosefs://``
     - Tencent Cloud GooseFS
     - ``GOOSEFS_MASTER_ADDR``
     - (optional) ``auth_type`` / ``auth_username`` properties

:class:`vortex.store.CosStore`
==============================

.. py:class:: vortex.store.CosStore(bucket, endpoint, *, secret_id=None, secret_key=None, root=None, disable_config_load=False)

   A Tencent Cloud COS object store, backed by OpenDAL. Construct it with explicit
   configuration and pass it to
   :func:`vortex.io.read_url` / :func:`vortex.io.write` via the ``store=`` argument,
   exactly like the built-in S3/Azure/GCS stores.

   The class is only available when Vortex is built with the ``opendal`` feature; on
   a default build, instantiating it raises :class:`ImportError`.

   :param bucket: COS bucket name (e.g. ``"my-bucket"``).
   :param endpoint: COS endpoint (e.g. ``"https://cos.ap-guangzhou.myqcloud.com"``).
   :param secret_id: Optional Tencent Cloud secret id. Maps to the ``TENCENTCLOUD_SECRET_ID``
       environment variable when unset.
   :param secret_key: Optional Tencent Cloud secret key. Maps to the
       ``TENCENTCLOUD_SECRET_KEY`` environment variable when unset.
   :param root: Optional key prefix applied to every operation.
   :param disable_config_load: When ``True``, disable OpenDAL's automatic config loading
       and rely only on the explicit configuration. Defaults to ``False``.

:class:`vortex.store.GoosefsStore`
==================================

.. py:class:: vortex.store.GoosefsStore(master_addr, *, root=None, block_size=None, chunk_size=None, write_type=None, auth_type=None, auth_username=None)

   A Tencent Cloud GooseFS object store, backed by OpenDAL. Construct it with explicit
   configuration and pass it to
   :func:`vortex.io.read_url` / :func:`vortex.io.write` via the ``store=`` argument,
   exactly like the built-in S3/Azure/GCS stores.

   The class is only available when Vortex is built with the ``opendal`` feature; on
   a default build, instantiating it raises :class:`ImportError`.

   :param master_addr: GooseFS master address(es). Single master:
       ``"10.0.0.1:9200"``. HA (comma-separated):
       ``"10.0.0.1:9200,10.0.0.2:9200,10.0.0.3:9200"``.
   :param root: Optional key prefix applied to every operation.
   :param block_size: Block size in bytes for new files (default: 64 MiB).
   :param chunk_size: Chunk size in bytes for streaming RPCs (default: 1 MiB).
   :param write_type: Default write type: ``"must_cache"``, ``"cache_through"``,
       ``"through"``, or ``"async_through"``.
   :param auth_type: Authentication type: ``"nosasl"`` or ``"simple"`` (default:
       ``"simple"``).
   :param auth_username: Authentication username (default: current OS user).

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

Reading from OSS
================

Alibaba Cloud OSS is reachable by URL. There is no standalone ``OssStore`` class yet, so
configuration comes from the environment variables OpenDAL's OSS builder reads
(``ALIBABA_CLOUD_ACCESS_KEY_ID``, ``ALIBABA_CLOUD_ACCESS_KEY_SECRET`` and ``OSS_ENDPOINT``):

.. code-block:: python

   import vortex as vx

   a = vx.io.read_url("oss://my-bucket/path/to/dataset.vortex")

Reading from GooseFS
====================

Pass a ``goosefs://`` URL directly. The master address is taken from the URL authority, or
from the ``GOOSEFS_MASTER_ADDR`` environment variable when the authority is empty:

.. code-block:: python

   import vortex as vx

   a = vx.io.read_url("goosefs://10.0.0.1:9200/path/to/dataset.vortex")

Or configure explicitly with :class:`~vortex.store.GoosefsStore` and pass it to
:func:`vortex.io.read_url` via ``store=``:

.. code-block:: python

   from vortex.io import read_url
   from vortex.store import GoosefsStore

   store = GoosefsStore(master_addr="10.0.0.1:9200")

   # When `store=` is supplied, the path is a key within the store, so the scheme and
   # master address are not part of the path passed to read_url.
   a = read_url("path/to/dataset.vortex", store=store)
