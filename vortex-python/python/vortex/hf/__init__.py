# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Hugging Face Hub integration for Vortex.

This module makes it possible to read Vortex files directly from the Hugging Face Hub
using ``hf://`` URLs, mirroring the URL convention used by ``huggingface_hub``'s
``HfFileSystem``::

    hf://datasets/{namespace}/{name}[@{revision}]/{path/in/repo}

URLs are translated into ranged HTTP reads against the Hub's ``resolve`` endpoint, so
Vortex's lazy scans (projection, predicate pushdown, row indices) only download the
bytes they need rather than whole files.

Examples
--------
Open a Vortex file hosted in a Hub dataset repository:

>>> import vortex as vx
>>> vxf = vx.open("hf://datasets/my-org/my-dataset/data/train.vortex") # doctest: +SKIP

Gated or private repositories are supported via Hugging Face tokens. Tokens are
resolved from the ``HF_TOKEN`` environment variable or the ``huggingface_hub`` token
cache by default, or may be passed explicitly:

>>> import vortex.hf
>>> vxf = vortex.hf.open("hf://datasets/my-org/private/data.vortex", token="hf_...") # doctest: +SKIP
"""

from __future__ import annotations

from ..file import VortexFile
from ..file import open as _open_file
from ..store import ClientConfig, RetryConfig
from ._resolve import (
    DEFAULT_ENDPOINT,
    DEFAULT_REVISION,
    HF_SCHEME,
    HFLocation,
    endpoint,
    http_store,
    resolve_url,
    store_and_path,
    token,
)


def open(
    url: str,
    *,
    token: str | None = None,
    client_options: ClientConfig | None = None,
    retry_config: RetryConfig | None = None,
    without_segment_cache: bool = False,
) -> VortexFile:
    """Lazily open a Vortex file from the Hugging Face Hub.

    This is equivalent to :func:`vortex.open` with an ``hf://`` URL, but additionally
    accepts an explicit ``token`` and HTTP client configuration.

    Parameters
    ----------
    url : :class:`str`
        An ``hf://`` URL such as ``hf://datasets/my-org/my-dataset/data/train.vortex``.
    token : :class:`str` | None
        A Hugging Face access token for gated or private repositories. Defaults to the
        ambient token resolved by :func:`token`.
    client_options :
        HTTP client options.
    retry_config :
        Retry configuration.
    without_segment_cache : :class:`bool`
        If true, disable the segment cache for this file.
    """
    store, path = store_and_path(url, token=token, client_options=client_options, retry_config=retry_config)
    return _open_file(path, store=store, without_segment_cache=without_segment_cache)


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_REVISION",
    "HF_SCHEME",
    "HFLocation",
    "endpoint",
    "http_store",
    "open",
    "resolve_url",
    "store_and_path",
    "token",
]
