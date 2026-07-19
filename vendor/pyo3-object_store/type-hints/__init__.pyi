# TODO: move to reusable types package
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

from ._aws import S3Config as S3Config
from ._aws import S3Credential as S3Credential
from ._aws import S3CredentialProvider as S3CredentialProvider
from ._aws import S3Store as S3Store
from ._azure import AzureAccessKey as AzureAccessKey
from ._azure import AzureBearerToken as AzureBearerToken
from ._azure import AzureConfig as AzureConfig
from ._azure import AzureCredential as AzureCredential
from ._azure import AzureCredentialProvider as AzureCredentialProvider
from ._azure import AzureSASToken as AzureSASToken
from ._azure import AzureStore as AzureStore
from ._client import ClientConfig as ClientConfig
from ._gcs import GCSConfig as GCSConfig
from ._gcs import GCSCredential as GCSCredential
from ._gcs import GCSCredentialProvider as GCSCredentialProvider
from ._gcs import GCSStore as GCSStore
from ._http import HTTPStore as HTTPStore
from ._retry import BackoffConfig as BackoffConfig
from ._retry import RetryConfig as RetryConfig

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 11):
    from typing import Self, Unpack
else:
    from typing_extensions import Self, Unpack

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
    credential_provider: Callable | None = None,
    **kwargs: Any,
) -> ObjectStore:
    """Easy construction of store by URL, identifying the relevant store.

    This will defer to a store-specific `from_url` constructor based on the provided
    `url`. E.g. passing `"s3://bucket/path"` will defer to
    [`S3Store.from_url`][obstore.store.S3Store.from_url].

    Supported formats:

    - `file:///path/to/my/file` -> [`LocalStore`][obstore.store.LocalStore]
    - `memory:///` -> [`MemoryStore`][obstore.store.MemoryStore]
    - `s3://bucket/path` -> [`S3Store`][obstore.store.S3Store] (also supports `s3a`)
    - `gs://bucket/path` -> [`GCSStore`][obstore.store.GCSStore]
    - `az://account/container/path` -> [`AzureStore`][obstore.store.AzureStore] (also
      supports `adl`, `azure`, `abfs`, `abfss`)
    - `http://mydomain/path` -> [`HTTPStore`][obstore.store.HTTPStore]
    - `https://mydomain/path` -> [`HTTPStore`][obstore.store.HTTPStore]

    There are also special cases for AWS and Azure for `https://{host?}/path` paths:

    - `dfs.core.windows.net`, `blob.core.windows.net`, `dfs.fabric.microsoft.com`,
      `blob.fabric.microsoft.com` -> [`AzureStore`][obstore.store.AzureStore]
    - `amazonaws.com` -> [`S3Store`][obstore.store.S3Store]
    - `r2.cloudflarestorage.com` -> [`S3Store`][obstore.store.S3Store]

    !!! note
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

class LocalStore:
    """An ObjectStore interface to local filesystem storage.

    Can optionally be created with a directory prefix.

    ```py
    from pathlib import Path

    store = LocalStore()
    store = LocalStore(prefix="/path/to/directory")
    store = LocalStore(prefix=Path("."))
    ```
    """

    def __init__(
        self,
        prefix: str | Path | None = None,
        *,
        automatic_cleanup: bool = False,
        mkdir: bool = False,
    ) -> None:
        """Create a new LocalStore.

        Args:
            prefix: Use the specified prefix applied to all paths. Defaults to `None`.

        Keyword Args:
            automatic_cleanup: if `True`, enables automatic cleanup of empty directories
                when deleting files. Defaults to False.
            mkdir: if `True` and `prefix` is not `None`, the directory at `prefix` will
                attempt to be created. Note that this root directory will not be cleaned
                up, even if `automatic_cleanup` is `True`.

        """
    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        automatic_cleanup: bool = False,
        mkdir: bool = False,
    ) -> Self:
        """Construct a new LocalStore from a `file://` URL.

        **Examples:**

        Construct a new store pointing to the root of your filesystem:
        ```py
        url = "file:///"
        store = LocalStore.from_url(url)
        ```

        Construct a new store with a directory prefix:
        ```py
        url = "file:///Users/kyle/"
        store = LocalStore.from_url(url)
        ```
        """

    def __eq__(self, value: object) -> bool: ...
    def __getnewargs_ex__(self): ...
    @property
    def prefix(self) -> Path | None:
        """Get the prefix applied to all operations in this store, if any."""

class MemoryStore:
    """A fully in-memory implementation of ObjectStore.

    Create a new in-memory store:
    ```py
    store = MemoryStore()
    ```
    """

    def __init__(self) -> None: ...

ObjectStore: TypeAlias = (
    AzureStore | GCSStore | HTTPStore | S3Store | LocalStore | MemoryStore
)
"""All supported ObjectStore implementations.

!!! warning "Not importable at runtime"

    To use this type hint in your code, import it within a `TYPE_CHECKING` block:

    ```py
    from __future__ import annotations
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from obstore.store import ObjectStore
    ```
"""
