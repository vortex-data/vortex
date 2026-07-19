import sys
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, Protocol, TypedDict

from ._client import ClientConfig
from ._retry import RetryConfig

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 11):
    from typing import Self, Unpack
else:
    from typing_extensions import Self, Unpack

class AzureConfig(TypedDict, total=False):
    """Configuration parameters for AzureStore.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import AzureConfig
        ```
    """

    account_name: str
    """The name of the azure storage account. (Required.)

    **Environment variable**: `AZURE_STORAGE_ACCOUNT_NAME`.
    """
    account_key: str
    """Master key for accessing storage account.

    **Environment variables**:

    - `AZURE_STORAGE_ACCOUNT_KEY`
    - `AZURE_STORAGE_ACCESS_KEY`
    - `AZURE_STORAGE_MASTER_KEY`
    """
    client_id: str
    """The client id for use in client secret or k8s federated credential flow.

    **Environment variables**:

    - `AZURE_STORAGE_CLIENT_ID`
    - `AZURE_CLIENT_ID`
    """
    client_secret: str
    """The client secret for use in client secret flow.

    **Environment variables**:

    - `AZURE_STORAGE_CLIENT_SECRET`
    - `AZURE_CLIENT_SECRET`
    """
    tenant_id: str
    """The tenant id for use in client secret or k8s federated credential flow.

    **Environment variables**:

    - `AZURE_STORAGE_TENANT_ID`
    - `AZURE_STORAGE_AUTHORITY_ID`
    - `AZURE_TENANT_ID`
    - `AZURE_AUTHORITY_ID`
    """
    authority_host: str
    """Sets an alternative authority host for OAuth based authorization.

    Defaults to `https://login.microsoftonline.com`.

    Common hosts for azure clouds are:

    - Azure China: `"https://login.chinacloudapi.cn"`
    - Azure Germany: `"https://login.microsoftonline.de"`
    - Azure Government: `"https://login.microsoftonline.us"`
    - Azure Public: `"https://login.microsoftonline.com"`

    **Environment variables**:

    - `AZURE_STORAGE_AUTHORITY_HOST`
    - `AZURE_AUTHORITY_HOST`
    """
    sas_key: str
    """
    Shared access signature.

    The signature is expected to be percent-encoded, `much `like they are provided in
    the azure storage explorer or azure portal.

    **Environment variables**:

    - `AZURE_STORAGE_SAS_KEY`
    - `AZURE_STORAGE_SAS_TOKEN`
    """
    token: str
    """A static bearer token to be used for authorizing requests.

    **Environment variable**: `AZURE_STORAGE_TOKEN`.
    """
    use_emulator: bool
    """Set if the Azure emulator should be used (defaults to `False`).

    **Environment variable**: `AZURE_STORAGE_USE_EMULATOR`.
    """
    use_fabric_endpoint: bool
    """Set if Microsoft Fabric url scheme should be used (defaults to `False`).

    When disabled the url scheme used is `https://{account}.blob.core.windows.net`.
    When enabled the url scheme used is `https://{account}.dfs.fabric.microsoft.com`.

    !!! note

        `endpoint` will take precedence over this option.
    """
    endpoint: str
    """Override the endpoint used to communicate with blob storage.

    Defaults to `https://{account}.blob.core.windows.net`.

    By default, only HTTPS schemes are enabled. To connect to an HTTP endpoint, enable
    `allow_http` in the client options.

    **Environment variables**:

    - `AZURE_STORAGE_ENDPOINT`
    - `AZURE_ENDPOINT`
    """
    msi_endpoint: str
    """Endpoint to request a imds managed identity token.

    **Environment variables**:

    - `AZURE_MSI_ENDPOINT`
    - `AZURE_IDENTITY_ENDPOINT`
    """
    object_id: str
    """Object id for use with managed identity authentication.

    **Environment variable**: `AZURE_OBJECT_ID`.
    """
    msi_resource_id: str
    """Msi resource id for use with managed identity authentication.

    **Environment variable**: `AZURE_MSI_RESOURCE_ID`.
    """
    federated_token_file: str
    """Sets a file path for acquiring azure federated identity token in k8s.

    Requires `client_id` and `tenant_id` to be set.

    **Environment variable**: `AZURE_FEDERATED_TOKEN_FILE`.
    """
    use_azure_cli: bool
    """Set if the Azure Cli should be used for acquiring access token.

    <https://learn.microsoft.com/en-us/cli/azure/account?view=azure-cli-latest#az-account-get-access-token>.

    **Environment variable**: `AZURE_USE_AZURE_CLI`.
    """
    skip_signature: bool
    """If enabled, `AzureStore` will not fetch credentials and will not sign requests.

    This can be useful when interacting with public containers.

    **Environment variable**: `AZURE_SKIP_SIGNATURE`.
    """
    container_name: str
    """Container name.

    **Environment variable**: `AZURE_CONTAINER_NAME`.
    """
    disable_tagging: bool
    """If set to `True` will ignore any tags provided to uploads.

    **Environment variable**: `AZURE_DISABLE_TAGGING`.
    """
    fabric_token_service_url: str
    """Service URL for Fabric OAuth2 authentication.

    **Environment variable**: `AZURE_FABRIC_TOKEN_SERVICE_URL`.
    """
    fabric_workload_host: str
    """Workload host for Fabric OAuth2 authentication.

    **Environment variable**: `AZURE_FABRIC_WORKLOAD_HOST`.
    """
    fabric_session_token: str
    """Session token for Fabric OAuth2 authentication.

    **Environment variable**: `AZURE_FABRIC_SESSION_TOKEN`.
    """
    fabric_cluster_identifier: str
    """Cluster identifier for Fabric OAuth2 authentication.

    **Environment variable**: `AZURE_FABRIC_CLUSTER_IDENTIFIER`.
    """

class AzureAccessKey(TypedDict):
    """A shared Azure Storage Account Key.

    <https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import AzureAccessKey
        ```
    """

    access_key: str
    """Access key value."""

    expires_at: datetime | None
    """Expiry datetime of credential. The datetime should have time zone set.

    If None, the credential will never expire.
    """

class AzureSASToken(TypedDict):
    """A shared access signature.

    <https://learn.microsoft.com/en-us/rest/api/storageservices/delegate-access-with-shared-access-signature>

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import AzureSASToken
        ```
    """

    sas_token: str | list[tuple[str, str]]
    """SAS token."""

    expires_at: datetime | None
    """Expiry datetime of credential. The datetime should have time zone set.

    If None, the credential will never expire.
    """

class AzureBearerToken(TypedDict):
    """An authorization token.

    <https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-azure-active-directory>

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import AzureBearerToken
        ```
    """

    token: str
    """Bearer token."""

    expires_at: datetime | None
    """Expiry datetime of credential. The datetime should have time zone set.

    If None, the credential will never expire.
    """

AzureCredential: TypeAlias = AzureAccessKey | AzureSASToken | AzureBearerToken
"""A type alias for supported azure credentials to be returned from `AzureCredentialProvider`.

!!! warning "Not importable at runtime"

    To use this type hint in your code, import it within a `TYPE_CHECKING` block:

    ```py
    from __future__ import annotations
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from obstore.store import AzureCredential
    ```
"""

class AzureCredentialProvider(Protocol):
    """A type hint for a synchronous or asynchronous callback to provide custom Azure credentials.

    This should be passed into the `credential_provider` parameter of `AzureStore`.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import AzureCredentialProvider
        ```
    """

    def __call__(self) -> AzureCredential | Coroutine[Any, Any, AzureCredential]:
        """Return an `AzureCredential`."""

class AzureStore:
    """Interface to a Microsoft Azure Blob Storage container.

    All constructors will check for environment variables. Refer to
    [`AzureConfig`][obstore.store.AzureConfig] for valid environment variables.
    """

    def __init__(  # type: ignore[misc] # Overlap between argument names and ** TypedDict items: "container_name"
        self,
        container_name: str | None = None,
        *,
        prefix: str | None = None,
        config: AzureConfig | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: AzureCredentialProvider | None = None,
        **kwargs: Unpack[AzureConfig],  # type: ignore # noqa: PGH003 (container_name key overlaps with positional arg)
    ) -> None:
        """Construct a new AzureStore.

        Args:
            container_name: the name of the container.

        Keyword Args:
            prefix: A prefix within the bucket to use for all operations.
            config: Azure Configuration. Values in this config will override values inferred from the url. Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom Azure credentials.
            kwargs: Azure configuration values. Supports the same values as `config`, but as named keyword args.

        Returns:
            AzureStore

        """

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        prefix: str | None = None,
        config: AzureConfig | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: AzureCredentialProvider | None = None,
        **kwargs: Unpack[AzureConfig],
    ) -> Self:
        """Construct a new AzureStore with values populated from a well-known storage URL.

        Any path on the URL will be assigned as the `prefix` for the store. So if you
        pass `https://<account>.blob.core.windows.net/<container>/path/to/directory`,
        the store will be created with a prefix of `path/to/directory`, and all further
        operations will use paths relative to that prefix.

        The supported url schemes are:

        - `abfs[s]://<container>/<path>` (according to [fsspec](https://github.com/fsspec/adlfs))
        - `abfs[s]://<file_system>@<account_name>.dfs.core.windows.net/<path>`
        - `abfs[s]://<file_system>@<account_name>.dfs.fabric.microsoft.com/<path>`
        - `az://<container>/<path>` (according to [fsspec](https://github.com/fsspec/adlfs))
        - `adl://<container>/<path>` (according to [fsspec](https://github.com/fsspec/adlfs))
        - `azure://<container>/<path>` (custom)
        - `https://<account>.dfs.core.windows.net`
        - `https://<account>.blob.core.windows.net`
        - `https://<account>.blob.core.windows.net/<container>`
        - `https://<account>.dfs.fabric.microsoft.com`
        - `https://<account>.dfs.fabric.microsoft.com/<container>`
        - `https://<account>.blob.fabric.microsoft.com`
        - `https://<account>.blob.fabric.microsoft.com/<container>`

        Args:
            url: well-known storage URL.

        Keyword Args:
            prefix: A prefix within the bucket to use for all operations.
            config: Azure Configuration. Values in this config will override values inferred from the url. Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom Azure credentials.
            kwargs: Azure configuration values. Supports the same values as `config`, but as named keyword args.

        Returns:
            AzureStore

        """

    def __eq__(self, value: object) -> bool: ...
    def __getnewargs_ex__(self): ...
    @property
    def prefix(self) -> str | None:
        """Get the prefix applied to all operations in this store, if any."""
    @property
    def config(self) -> AzureConfig:
        """Get the underlying Azure config parameters."""
    @property
    def client_options(self) -> ClientConfig | None:
        """Get the store's client configuration."""
    @property
    def credential_provider(self) -> AzureCredentialProvider | None:
        """Get the store's credential provider."""
    @property
    def retry_config(self) -> RetryConfig | None:
        """Get the store's retry configuration."""
