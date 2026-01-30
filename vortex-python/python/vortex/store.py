# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import TYPE_CHECKING

from ._lib.store import (  # pyright: ignore[reportMissingModuleSource]
    AzureStore,
    GCSStore,
    HTTPStore,
    LocalStore,
    MemoryStore,
    S3Store,
    from_url,
)

if TYPE_CHECKING:
    from ._lib.store import (  # pyright: ignore[reportMissingModuleSource]
        AzureAccessKey,
        AzureBearerToken,
        AzureConfig,
        AzureCredential,
        AzureCredentialProvider,
        AzureSASToken,
        BackoffConfig,
        ClientConfig,
        GCSConfig,
        GCSCredential,
        GCSCredentialProvider,
        ObjectStore,
        RetryConfig,
        S3Config,
        S3Credential,
        S3CredentialProvider,
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
