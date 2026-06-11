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

Use :func:`register_datasets` to teach the :mod:`datasets` library how to load
``.vortex`` files with ``datasets.load_dataset``:

>>> import vortex.hf
>>> vortex.hf.register_datasets() # doctest: +SKIP
>>> import datasets
>>> ds = datasets.load_dataset("vortex", data_files="data/*.vortex") # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pyarrow as pa

from ..file import VortexFile
from ..file import open as _open_file
from ..io import write as _write
from ..store import ClientConfig, ObjectStore, RetryConfig
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

if TYPE_CHECKING:
    import datasets as hf_datasets


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


def dataset_to_vortex(dataset: hf_datasets.Dataset, path: str, *, store: ObjectStore | None = None) -> None:
    """Write a Hugging Face :class:`datasets.Dataset` to a Vortex file.

    The dataset's backing Arrow table is written directly, preserving the schema.

    Examples
    --------
    >>> import datasets # doctest: +SKIP
    >>> ds = datasets.load_dataset("my-org/my-dataset", split="train") # doctest: +SKIP
    >>> import vortex.hf
    >>> vortex.hf.dataset_to_vortex(ds, "train.vortex") # doctest: +SKIP
    """
    table = dataset.data
    # datasets wraps the backing pyarrow.Table in datasets.table.Table.
    arrow_table = cast(pa.Table, getattr(table, "table", table))
    _write(arrow_table, path, store=store)


def register_datasets() -> None:
    """Register the Vortex builder with the :mod:`datasets` library.

    After calling this, ``datasets.load_dataset`` understands the ``"vortex"`` builder
    name as well as ``.vortex`` data files::

        import vortex.hf
        vortex.hf.register_datasets()

        import datasets
        ds = datasets.load_dataset("vortex", data_files={"train": "data/*.vortex"})

    Requires the ``datasets`` package (``pip install vortex-data[huggingface]``).
    """
    import inspect

    import datasets.packaged_modules as packaged_modules

    from . import builder as builder_module

    module_hash = packaged_modules._hash_python_lines(  # pyright: ignore[reportPrivateUsage]
        inspect.getsource(builder_module).splitlines()
    )
    packaged_modules._PACKAGED_DATASETS_MODULES["vortex"] = (  # pyright: ignore[reportPrivateUsage]
        builder_module.__name__,
        module_hash,
    )
    packaged_modules._EXTENSION_TO_MODULE[".vortex"] = ("vortex", {})  # pyright: ignore[reportPrivateUsage]
    # Older and newer versions of `datasets` index slightly different metadata tables.
    if hasattr(packaged_modules, "_MODULE_TO_EXTENSIONS"):
        packaged_modules._MODULE_TO_EXTENSIONS["vortex"] = [".vortex"]  # pyright: ignore[reportPrivateUsage]
    if hasattr(packaged_modules, "_MODULE_TO_METADATA_FILE_NAMES"):
        packaged_modules._MODULE_TO_METADATA_FILE_NAMES["vortex"] = []  # pyright: ignore[reportPrivateUsage]


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_REVISION",
    "HF_SCHEME",
    "HFLocation",
    "dataset_to_vortex",
    "endpoint",
    "http_store",
    "open",
    "register_datasets",
    "resolve_url",
    "store_and_path",
    "token",
]
