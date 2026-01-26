# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 Development Seed

from collections.abc import Coroutine
from datetime import datetime
from typing import Any, Protocol, Self, TypedDict, Unpack

from ._client import ClientConfig
from ._retry import RetryConfig

class GCSConfig(TypedDict, total=False):
    """Configuration parameters for GCSStore.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import GCSConfig
        ```
    """

    service_account: str
    """Path to the service account file.

    This or `service_account_key` must be set.

    Example value `"/tmp/gcs.json"`. Example contents of `gcs.json`:

    ```json
    {
       "gcs_base_url": "https://localhost:4443",
       "disable_oauth": true,
       "client_email": "",
       "private_key": ""
    }
    ```

    **Environment variables**:

    - `GOOGLE_SERVICE_ACCOUNT`
    - `GOOGLE_SERVICE_ACCOUNT_PATH`
    """

    service_account_key: str
    """The serialized service account key.

    The service account must be in the JSON format. This or `with_service_account_path`
    must be set.

    **Environment variable**: `GOOGLE_SERVICE_ACCOUNT_KEY`.
    """

    bucket: str
    """Bucket name. (required)

    **Environment variables**:

    - `GOOGLE_BUCKET`
    - `GOOGLE_BUCKET_NAME`
    """

    application_credentials: str
    """Application credentials path.

    See <https://cloud.google.com/docs/authentication/provide-credentials-adc>.

    **Environment variable**: `GOOGLE_APPLICATION_CREDENTIALS`.
    """

    skip_signature: bool
    """If `True`, GCSStore will not fetch credentials and will not sign requests.

    This can be useful when interacting with public GCS buckets that deny authorized requests.

    **Environment variable**: `GOOGLE_SKIP_SIGNATURE`.
    """

class GCSCredential(TypedDict):
    """A Google Cloud Storage Credential.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import GCSCredential
        ```
    """

    token: str
    """An HTTP bearer token."""

    expires_at: datetime | None
    """Expiry datetime of credential. The datetime should have time zone set.

    If None, the credential will never expire.
    """

class GCSCredentialProvider(Protocol):
    """A type hint for a synchronous or asynchronous callback to provide custom Google Cloud Storage credentials.

    This should be passed into the `credential_provider` parameter of `GCSStore`.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import GCSCredentialProvider
        ```
    """

    def __call__(self) -> GCSCredential | Coroutine[Any, Any, GCSCredential]:
        """Return a `GCSCredential`."""

class GCSStore:
    """Interface to Google Cloud Storage.

    All constructors will check for environment variables. Refer to
    [`GCSConfig`][vortex.store.GCSConfig] for valid environment variables.

    If no credentials are explicitly provided, they will be sourced from the environment
    as documented
    [here](https://cloud.google.com/docs/authentication/application-default-credentials).
    """

    def __init__(  # type: ignore[misc] # Overlap between argument names and ** TypedDict items: "bucket"
        self,
        bucket: str | None = None,
        *,
        prefix: str | None = None,
        config: GCSConfig | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: GCSCredentialProvider | None = None,
        **kwargs: Unpack[GCSConfig],  # type: ignore # noqa: PGH003 (bucket key overlaps with positional arg)
    ) -> None:
        """Construct a new GCSStore.

        Args:
            bucket: The GCS bucket to use.

        Keyword Args:
            prefix: A prefix within the bucket to use for all operations.
            config: GCS Configuration. Values in this config will override values inferred from the
            environment. Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom Google credentials.
            kwargs: GCS configuration values. Supports the same values as `config`,
            but as named keyword args.

        Returns:
            GCSStore

        """

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        prefix: str | None = None,
        config: GCSConfig | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: GCSCredentialProvider | None = None,
        **kwargs: Unpack[GCSConfig],
    ) -> Self:
        """Construct a new GCSStore with values populated from a well-known storage URL.

        Any path on the URL will be assigned as the `prefix` for the store. So if you
        pass `gs://<bucket>/path/to/directory`, the store will be created with a prefix
        of `path/to/directory`, and all further operations will use paths relative to
        that prefix.

        The supported url schemes are:

        - `gs://<bucket>/<path>`

        Args:
            url: well-known storage URL.

        Keyword Args:
            prefix: A prefix within the bucket to use for all operations.
            config: GCS Configuration. Values in this config will override values inferred from the
            url. Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom Google credentials.
            kwargs: GCS configuration values. Supports the same values as `config`, but as named keyword
            args.

        Returns:
            GCSStore

        """

    def __eq__(self, value: object) -> bool: ...
    def __getnewargs_ex__(self): ...
    @property
    def prefix(self) -> str | None:
        """Get the prefix applied to all operations in this store, if any."""
    @property
    def config(self) -> GCSConfig:
        """Get the underlying GCS config parameters."""
    @property
    def client_options(self) -> ClientConfig | None:
        """Get the store's client configuration."""
    @property
    def credential_provider(self) -> GCSCredentialProvider | None:
        """Get the store's credential provider."""
    @property
    def retry_config(self) -> RetryConfig | None:
        """Get the store's retry configuration."""
