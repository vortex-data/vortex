==================
OpenDAL (COS, OSS)
==================

Vortex can read from and write to Tencent Cloud COS and Alibaba Cloud OSS through
`OpenDAL <https://opendal.apache.org/>`_, which provides native service support.

These stores are available only when Vortex is built with the ``opendal`` feature
(e.g. ``maturin develop --features opendal`` or ``cargo build -p vortex-jni --features opendal``).

.. list-table::
   :header-rows: 1

   * - Scheme
     - Service
     - Endpoint variable
     - Credential variables
   * - ``cos://``
     - Tencent Cloud COS
     - ``COS_ENDPOINT``
     - ``TENCENTCLOUD_SECRET_ID``, ``TENCENTCLOUD_SECRET_KEY``
   * - ``oss://``
     - Alibaba Cloud OSS
     - ``OSS_ENDPOINT``
     - ``ALIBABA_CLOUD_ACCESS_KEY_ID``, ``ALIBABA_CLOUD_ACCESS_KEY_SECRET``

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
