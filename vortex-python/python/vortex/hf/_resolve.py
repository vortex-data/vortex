# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Translation of ``hf://`` URLs into HTTP stores against the Hugging Face Hub.

This module only depends on the store layer so that :mod:`vortex.file` and
:mod:`vortex.store` can use it without import cycles. The user-facing API is re-exported
from :mod:`vortex.hf`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from ..store._client import ClientConfig
from ..store._http import HTTPStore
from ..store._retry import RetryConfig

HF_SCHEME = "hf://"
"""URL scheme for Hugging Face Hub locations."""

DEFAULT_ENDPOINT = "https://huggingface.co"
"""The default Hugging Face Hub endpoint."""

DEFAULT_REVISION = "main"
"""The revision used when an ``hf://`` URL does not carry an ``@revision`` suffix."""

_PREFIX_TO_REPO_TYPE = {"datasets": "dataset", "spaces": "space"}
_REPO_TYPE_TO_PREFIX = {"dataset": "datasets/", "space": "spaces/", "model": ""}

_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def endpoint() -> str:
    """The Hugging Face Hub endpoint, honoring the ``HF_ENDPOINT`` environment variable."""
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def token(explicit: str | None = None) -> str | None:
    """Resolve a Hugging Face access token, or None if no token is configured.

    Resolution order matches ``huggingface_hub``:

    1. The ``explicit`` argument, if provided.
    2. The ``HF_TOKEN``, ``HUGGING_FACE_HUB_TOKEN``, or ``HUGGINGFACE_TOKEN``
       environment variables.
    3. The token cache file at ``$HF_TOKEN_PATH``, falling back to
       ``$HF_HOME/token`` and then ``~/.cache/huggingface/token``.
    """
    if explicit:
        return explicit
    for var in _TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path is None:
        hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        token_path = os.path.join(hf_home, "token")
    try:
        return Path(token_path).read_text().strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class HFLocation:
    """A parsed ``hf://`` URL identifying a file or directory in a Hub repository."""

    repo_id: str
    """The repository id, e.g. ``"my-org/my-dataset"``."""
    path: str = ""
    """The path of the file or directory within the repository."""
    repo_type: str = "dataset"
    """One of ``"dataset"``, ``"model"``, or ``"space"``."""
    revision: str = DEFAULT_REVISION
    """A branch name, tag, or commit SHA, e.g. ``"main"`` or ``"refs/convert/parquet"``."""

    @classmethod
    def parse(cls, url: str) -> HFLocation:
        """Parse an ``hf://`` URL.

        The accepted format follows the ``HfFileSystem`` convention::

            hf://[datasets/|spaces/]{namespace}/{name}[@{revision}]/{path/in/repo}

        Repositories without a ``datasets/`` or ``spaces/`` prefix are treated as model
        repositories. Repository ids must be fully qualified as ``namespace/name``; the
        optional revision may be percent-encoded (e.g. ``@refs%2Fconvert%2Fparquet``).

        Examples
        --------
        >>> from vortex.hf import HFLocation
        >>> HFLocation.parse("hf://datasets/my-org/my-data@v1.0/dir/file.vortex")
        HFLocation(repo_id='my-org/my-data', path='dir/file.vortex', repo_type='dataset', revision='v1.0')
        """
        if not url.startswith(HF_SCHEME):
            raise ValueError(f"not an {HF_SCHEME} URL: {url!r}")
        segments = [s for s in url[len(HF_SCHEME) :].split("/") if s]

        repo_type = "model"
        if segments and segments[0] in _PREFIX_TO_REPO_TYPE:
            repo_type = _PREFIX_TO_REPO_TYPE[segments[0]]
            segments = segments[1:]

        if len(segments) < 2:
            raise ValueError(
                f"invalid Hugging Face URL {url!r}: expected "
                f"hf://[datasets/|spaces/]namespace/name[@revision]/path/in/repo"
            )

        namespace = segments[0]
        name, _, revision = segments[1].partition("@")
        if not namespace or not name or ("@" in segments[1] and not revision):
            raise ValueError(f"invalid Hugging Face URL {url!r}: empty repository namespace, name, or revision")

        return cls(
            repo_id=f"{namespace}/{name}",
            path="/".join(segments[2:]),
            repo_type=repo_type,
            revision=unquote(revision) if revision else DEFAULT_REVISION,
        )

    def resolve_root(self, hub_endpoint: str | None = None) -> str:
        """The HTTPS URL of this repository revision's ``resolve`` root."""
        prefix = _REPO_TYPE_TO_PREFIX[self.repo_type]
        base = hub_endpoint.rstrip("/") if hub_endpoint is not None else endpoint()
        return f"{base}/{prefix}{self.repo_id}/resolve/{quote(self.revision, safe='')}"

    def resolve_url(self, hub_endpoint: str | None = None) -> str:
        """The full HTTPS ``resolve`` URL for this location, including its path."""
        root = self.resolve_root(hub_endpoint)
        return f"{root}/{self.path}" if self.path else root


def resolve_url(url: str, hub_endpoint: str | None = None) -> str:
    """Translate an ``hf://`` URL to the HTTPS URL it resolves to on the Hub.

    Examples
    --------
    >>> from vortex.hf import resolve_url
    >>> resolve_url("hf://datasets/my-org/my-data/file.vortex")
    'https://huggingface.co/datasets/my-org/my-data/resolve/main/file.vortex'
    """
    return HFLocation.parse(url).resolve_url(hub_endpoint)


def _client_options(url: str, client_options: ClientConfig | None, explicit_token: str | None) -> ClientConfig | None:
    """Build client options for a Hub store: attach auth, and permit non-TLS endpoints.

    A plain-HTTP endpoint can only arise from an explicit ``HF_ENDPOINT`` override (e.g.
    a local mirror or test fixture), so ``allow_http`` is enabled for it by default.
    """
    options = ClientConfig()
    if client_options is not None:
        options.update(client_options)
    if url.startswith("http://"):
        options.setdefault("allow_http", True)
    resolved = token(explicit_token)
    if resolved is not None:
        headers = {str(k): v for k, v in options.get("default_headers", {}).items()}
        headers.setdefault("authorization", f"Bearer {resolved}")
        options["default_headers"] = headers  # pyright: ignore
    return options or None


def http_store(
    url: str,
    *,
    token: str | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
) -> HTTPStore:
    """Construct an :class:`~vortex.store.HTTPStore` rooted at an ``hf://`` URL.

    The store is rooted at the Hub ``resolve`` URL for the location, so paths passed to
    subsequent operations are relative to the ``hf://`` URL's path.

    Parameters
    ----------
    url : :class:`str`
        An ``hf://`` URL, see :meth:`HFLocation.parse` for the accepted format.
    token : :class:`str` | None
        A Hugging Face access token for gated or private repositories. Defaults to the
        ambient token resolved by :func:`token`.
    client_options :
        HTTP client options. An ``authorization`` header is added when a token resolves.
    retry_config :
        Retry configuration.
    """
    resolved = HFLocation.parse(url).resolve_url()
    return HTTPStore(
        resolved,
        client_options=_client_options(resolved, client_options, token),
        retry_config=retry_config,
    )


def store_and_path(
    url: str,
    *,
    token: str | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
) -> tuple[HTTPStore, str]:
    """Split an ``hf://`` file URL into an :class:`~vortex.store.HTTPStore` and a relative path.

    The store is rooted at the repository revision's ``resolve`` root, and the returned
    path locates the file within the repository.
    """
    location = HFLocation.parse(url)
    if not location.path:
        raise ValueError(f"Hugging Face URL {url!r} does not contain a file path within the repository")
    root = location.resolve_root()
    store = HTTPStore(
        root,
        client_options=_client_options(root, client_options, token),
        retry_config=retry_config,
    )
    return store, location.path
