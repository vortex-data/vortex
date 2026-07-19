# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""OpenDAL-backed object store for Tencent Cloud COS.

This store is only available when Vortex is built with the ``opendal`` feature. Unlike the
URL-based resolution (``cos://``), this class is a concrete, standalone
``ObjectStore`` object that you can build once and pass directly to
:func:`vortex.io.read_url` / :func:`vortex.io.write` via the ``store=`` argument::

    from vortex.store import CosStore
    from vortex.io import read_url

    store = CosStore(
        bucket="my-bucket",
        endpoint="https://cos.ap-guangzhou.myqcloud.com",
        secret_id="AKID...",
        secret_key="...",
    )
    array = read_url("cos://my-bucket/data.vortex", store=store)

Credentials may also be supplied via the environment variables that OpenDAL's builders read
automatically (``TENCENTCLOUD_SECRET_ID`` / ``TENCENTCLOUD_SECRET_KEY`` and ``COS_ENDPOINT``);
any value passed to the constructor takes precedence over the environment.
"""

from __future__ import annotations

from vortex._lib import CosStore

__all__ = ["CosStore"]


class _OpenDALStore:
    """Base helper that projects keyword configuration into OpenDAL environment variables."""

    # (our kwarg name) -> (env var OpenDAL's builder reads)
    _ENV_MAP: dict[str, str]

    def __init__(self, **kwargs: Any) -> None:
        self._config: dict[str, str] = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            self._config[key] = str(value)

    def apply(self) -> None:
        """Export this store's configuration into the process environment.

        After calling :meth:`apply`, ``cos://`` / ``oss://`` URLs resolve using these values.
        """
        for key, env_var in self._ENV_MAP.items():
            if key in self._config:
                os.environ[env_var] = self._config[key]


class CosStore(_OpenDALStore):
    """Configuration helper for Tencent Cloud COS, resolved from ``cos://`` URLs.

    Keyword Args:
        bucket: COS bucket name.
        endpoint: COS endpoint, e.g. ``https://cos.ap-guangzhou.myqcloud.com``.
        secret_id: Tencent Cloud secret id (mapped to ``TENCENTCLOUD_SECRET_ID``).
        secret_key: Tencent Cloud secret key (mapped to ``TENCENTCLOUD_SECRET_KEY``).
        root: Optional root prefix applied to all operations.
    """

        _ENV_MAP = {
        "secret_id": "TENCENTCLOUD_SECRET_ID",
        "secret_key": "TENCENTCLOUD_SECRET_KEY",
        "endpoint": "COS_ENDPOINT",
    }
