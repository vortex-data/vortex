# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 Development Seed

from typing import Self

from typing_extensions import override

from .._lib import store as _store  # pyright: ignore[reportMissingModuleSource]
from ._client import ClientConfig
from ._retry import RetryConfig


class HTTPStore(_store.HTTPStore):
    """Configure a connection to a generic HTTP server."""

    def __new__(
        cls,
        url: str,
        *,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """Construct a new HTTPStore from a URL.

        Any path on the URL will be assigned as the ``prefix`` for the store. So if you
        pass ``https://example.com/path/to/directory``, the store will be created with a
        prefix of ``path/to/directory``, and all further operations will use paths
        relative to that prefix.

        Args:
            url: The base URL to use for the store.

        Keyword Args:
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.

        Returns:
            HTTPStore

        """
        return super().__new__(url, client_options=client_options, retry_config=retry_config)

    @override
    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> Self:
        """Construct a new HTTPStore from a URL.

        This is an alias of the :class:`~vortex.store.HTTPStore` constructor.
        """
        return super(cls).from_url(url, client_options=client_options, retry_config=retry_config)

    @override
    def __eq__(self, value: object) -> bool:
        return super().__eq__(value)

    @override
    def __getnewargs_ex__(self):  # pyright: ignore[reportUnknownParameterType]
        return super().__getnewargs_ex__()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    @property
    @override
    def url(self) -> str:
        """Get the base url of this store."""
        return super().url

    @property
    @override
    def client_options(self) -> ClientConfig | None:
        """Get the store's client configuration."""
        return super().client_options

    @property
    @override
    def retry_config(self) -> RetryConfig | None:
        """Get the store's retry configuration."""
        return super().retry_config
