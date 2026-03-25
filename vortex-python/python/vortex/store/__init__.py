# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Callable
from typing import TypeAlias, Unpack, overload

from .._lib import store as _store  # pyright: ignore[reportMissingModuleSource]
from ._aws import *
from ._azure import *
from ._client import *
from ._gcs import *
from ._http import *
from ._retry import *
from ._local import *
from ._memory import *


ObjectStore: TypeAlias = AzureStore | GCSStore | HTTPStore | S3Store | LocalStore | MemoryStore
"""All supported ObjectStore implementations."""


@overload
def from_url(
    url: str,
    *,
    config: S3Config | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
    credential_provider: S3CredentialProvider | None = None,
    **kwargs: Unpack[S3Config],
) -> ObjectStore: ...
@overload
def from_url(
    url: str,
    *,
    config: GCSConfig | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
    credential_provider: GCSCredentialProvider | None = None,
    **kwargs: Unpack[GCSConfig],
) -> ObjectStore: ...
@overload
def from_url(
    url: str,
    *,
    config: AzureConfig | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
    credential_provider: AzureCredentialProvider | None = None,
    **kwargs: Unpack[AzureConfig],
) -> ObjectStore: ...
@overload
def from_url(
    url: str,
    *,
    config: None = None,
    client_options: None = None,
    retry_config: None = None,
    automatic_cleanup: bool = False,
    mkdir: bool = False,
) -> ObjectStore: ...
def from_url(  # type: ignore[misc] # docstring in pyi file
    url: str,
    *,
    config: S3Config | GCSConfig | AzureConfig | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
    credential_provider: Callable[..., object] | None = None,
    **kwargs: object,
) -> ObjectStore:
    """Easy construction of store by URL, identifying the relevant store.

    This will defer to a store-specific ``from_url`` constructor based on the provided
    ``url``. E.g. passing ``"s3://bucket/path"`` will defer to
    :meth:`S3Store.from_url <vortex.store.S3Store.from_url>`.

    Supported formats:

    - ``file:///path/to/my/file`` -> :class:`~vortex.store.LocalStore`
    - ``memory:///`` -> :class:`~vortex.store.MemoryStore`
    - ``s3://bucket/path`` -> :class:`~vortex.store.S3Store` (also supports ``s3a``)
    - ``gs://bucket/path`` -> :class:`~vortex.store.GCSStore`
    - ``az://account/container/path`` -> :class:`~vortex.store.AzureStore` (also
      supports ``adl``, ``azure``, ``abfs``, ``abfss``)
    - ``http://mydomain/path`` -> :class:`~vortex.store.HTTPStore`
    - ``https://mydomain/path`` -> :class:`~vortex.store.HTTPStore`

    There are also special cases for AWS and Azure for ``https://{host?}/path`` paths:

    - ``dfs.core.windows.net``, ``blob.core.windows.net``, ``dfs.fabric.microsoft.com``,
      ``blob.fabric.microsoft.com`` -> :class:`~vortex.store.AzureStore`
    - ``amazonaws.com`` -> :class:`~vortex.store.S3Store`
    - ``r2.cloudflarestorage.com`` -> :class:`~vortex.store.S3Store`

    .. note::

        For best static typing, use the constructors on individual store classes
        directly.

    Args:
        url: well-known storage URL.

    Keyword Args:
        config: per-store Configuration. Values in this config will override values
            inferred from the url. Defaults to None.
        client_options: HTTP Client options. Defaults to None.
        retry_config: Retry configuration. Defaults to None.
        credential_provider: A callback to provide custom credentials to the underlying store classes.
        kwargs: per-store configuration passed down to store-specific builders.

    """
    return _store.from_url(
        url,
        config=config,
        client_options=client_options,
        retry_config=retry_config,
        credential_provider=credential_provider,
        **kwargs,
    )


__all__ = [
    # Azure
    "AzureAccessKey",
    "AzureBearerToken",
    "AzureConfig",
    "AzureCredential",
    "AzureCredentialProvider",
    "AzureSASToken",
    "AzureStore",
    # Client
    "BackoffConfig",
    "ClientConfig",
    "RetryConfig",
    # GCS
    "GCSConfig",
    "GCSCredential",
    "GCSCredentialProvider",
    "GCSStore",
    # HTTP
    "HTTPStore",
    # Local
    "LocalStore",
    "MemoryStore",
    # S3
    "S3Config",
    "S3Credential",
    "S3CredentialProvider",
    "S3Store",
    # Utility
    "from_url",
    "ObjectStore",
]
